# 🛠️ IrisAgent End-to-End Setup Guide

This guide walks a new person through deploying the IrisAgent server on an AWS Graviton instance and running the client on a local laptop.

**Contents**

- [Part 1: Set up the AWS Graviton server (cloud)](#part-1-set-up-the-aws-graviton-server-cloud)
- [Part 2: Set up the laptop client (local)](#part-2-set-up-the-laptop-client-local)
- [Troubleshooting](#troubleshooting)

---

## Part 1: Set up the AWS Graviton server (cloud)

### Step 1: Launch an AWS EC2 Graviton instance

1. Log in to the AWS Management Console and go to **EC2**.
2. Click **Launch instance**.
3. Configure the instance:
   - **Name:** `IrisAgent-Server`
   - **AMI:** Ubuntu 22.04 LTS or 24.04 LTS. **Choose the ARM64 version.** Graviton uses ARM, so x86 images will not run.
   - **Instance type:** `c7g.large` (AWS Graviton3)
   - **Key pair:** Create or select a `.pem` SSH key.
   - **Network settings:**
     - Allow SSH (port 22) from **your IP address only**.
     - Add a custom TCP rule for port **5000** (Flask server) from **your IP address only**.
     - > ⚠️ Avoid `0.0.0.0/0` (anywhere). It exposes the server to the whole internet. Use your own IP, or a VPN.
   - **Storage:** At least 30 GB gp3.
4. Click **Launch instance**.

> **Note:** To reproduce the COOL benchmarks from the report, use the **COOL Graviton3 AMI** (`COOL-graviton3-v3-prod-xcqopwa7u5zdu`) instead of the standard Ubuntu AMI. The steps below still apply.

### Step 2: Connect to the instance

From your local terminal:

```bash
ssh -i "your-key.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
```

### Step 3: Install system dependencies and the Python environment

After logging in to the EC2 instance:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git build-essential
```

### Step 4: Clone the code and upload the ONNX models

1. Clone the project:

   ```bash
   git clone https://github.com/your-username/IrisAgent.git
   cd IrisAgent
   ```

2. Upload the model files from your local machine with `scp`:

   ```bash
   scp -i "your-key.pem" eye_gaze_blink_mouthopen_best.onnx ubuntu@<YOUR_EC2_PUBLIC_IP>:/home/ubuntu/IrisAgent/
   scp -i "your-key.pem" eye_gaze_model_2.onnx ubuntu@<YOUR_EC2_PUBLIC_IP>:/home/ubuntu/IrisAgent/
   ```

### Step 5: Install Python requirements

```bash
pip install -r requirements-server.txt
```

> If you see a "externally-managed-environment" error, create a virtual environment first:
> ```bash
> python3 -m venv venv
> source venv/bin/activate
> pip install -r requirements-server.txt
> ```

### Step 6: Configure environment variables

Set the variables in your terminal session:

```bash
export AWS_REGION="ap-south-1"
export<ACCESS KEYS>
export IOT_ENDPOINT="<your-endpoint>-ats.iot.ap-south-1.amazonaws.com"
export DYNAMO_TABLE_NAME="IrisAgentLogs"
export GEMINI_API_KEY="<your-google-api-key>"
```

> **Security tips**
> - **Don't commit keys to Git.** Keep them out of the repository.
> - **Prefer an IAM role over stored keys.** Attach an IAM role to the EC2 instance that allows publishing to SNS, IoT Core, and writing to DynamoDB, so you don't need AWS access keys on the server.
> - **Use a secrets file or a `.env` file** that is listed in `.gitignore` instead of adding exports to `~/.bashrc`, which is easy to leak.
> - Keep `GEMINI_API_KEY` in the same protected location.

### Step 7: Run the server

```bash
python3 server_iris_gemini_bedrockremoved_v2.py
```

Successful startup shows:

- Log lines confirming the ONNX models loaded with Graviton optimizations
- The server running on `http://0.0.0.0:5000`

---

## Part 2: Set up the laptop client (local)

### Step 1: Clone the repository

On your laptop:

```bash
git clone https://github.com/your-username/IrisAgent.git
cd IrisAgent
```

### Step 2: Install client dependencies

```bash
pip install -r requirements-client.txt
```

This installs the packages for video rendering and audio playback.

### Step 3: Connect and verify the cameras

Connect two webcams:

| Camera | Points at | Default index |
|---|---|---|
| **Inward camera** | Your face and eyes | `0` |
| **Outward camera** | Your room or desk | `1` |

To check the camera indices, open a short Python shell and try reading from each index, or check your system's camera settings.

### Step 4: Point the client at the server

Open `client_iris.py` in a text editor and set `SERVER_BASE` to your EC2 public IP:

```python
SERVER_BASE = os.getenv("IRIS_SERVER", "http://<YOUR_EC2_PUBLIC_IP>:5000")
```

You can also set the address without editing the file:

```bash
export IRIS_SERVER="http://<YOUR_EC2_PUBLIC_IP>:5000"
```

### Step 5: Run the client

```bash
python client_iris.py
```

What you should see:

- The **IrisAgent** window opens and shows the inward camera feed with real-time blink and gaze detection boxes.
- When you enter **Agentic Vision Mode** (press **2**), an **Environment View** window opens. You can then control the system hands-free with your eyes.

---

## Troubleshooting

| Problem | What to check |
|---|---|
| `exec format error` or models won't load | You launched an x86 AMI. Relaunch with an **ARM64** (Graviton) image. |
| Client can't connect to the server | Confirm the EC2 instance is running, the port 5000 rule allows your IP, and `IRIS_SERVER` has the correct public IP. |
| Public IP changed after restart | EC2 public IPs change when an instance stops and starts. Use an Elastic IP or update `IRIS_SERVER`. |
| Camera doesn't open or shows the wrong view | Change the camera index (try `0` and `1` swapped, or higher values). |
| AWS errors (SNS, DynamoDB, IoT) | Check the IAM role on the instance and the region (`ap-south-1`). |
| Port 5000 blocked | Check the security group rule and that the server is listening on `0.0.0.0:5000`. |
| Instance costs | Stop the instance when not in use: `aws ec2 stop-instances --instance-ids <id> --region ap-south-1` |

> **Remember:** Stop the EC2 instance after the demo. This reduces cost and attack surface, as described in the project report.
