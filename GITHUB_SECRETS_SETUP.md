<<<<<<< HEAD
# GitHub Repository Secrets Configuration
## 24/7 Cloud Learning Pipeline Setup

### Required Secrets

You need to configure the following secrets in your GitHub repository:

**Repository Settings > Secrets and variables > Actions > New repository secret**

#### 1. AWS Credentials & Configuration
```
AWS_ACCESS_KEY_ID
Value: [Your AWS access key from IAM user/role]

AWS_SECRET_ACCESS_KEY  
Value: [Your AWS secret access key]

AWS_REGION
Value: us-east-1 (or your preferred region)
```

#### 2. S3 Storage Configuration
```
S3_BUCKET
Value: [Your S3 bucket name, e.g., "topstep-bot-ml-models"]

CDN_BASE_URL
Value: [CloudFront or S3 public URL, e.g., "https://d1234567890.cloudfront.net"]
```

#### 3. Security (HMAC Manifest Signing)
```
MANIFEST_HMAC_KEY
Value: [Generate a random 64-char hex string for manifest integrity]
```
### AWS Setup Requirements

#### S3 Bucket Policy (Replace `your-bucket-name`)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowGitHubActions",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::YOUR-ACCOUNT-ID:user/github-actions-user"
      },
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    },
    {
      "Sid": "AllowPublicRead",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::your-bucket-name/models/*"
    }
  ]
}
```

#### IAM User Policy for GitHub Actions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
```

### Generate HMAC Key
```bash
# Generate secure 64-character hex key
openssl rand -hex 32
```

### How to get AWS credentials:
=======
# 🔐 GitHub Secrets Setup Guide

## **Required Secrets for Cloud Learning**

Go to your GitHub repository: `https://github.com/kevinsuero072897-collab/trading-bot-c-/settings/secrets/actions`

Add these secrets:

### **AWS Credentials:**
```
Name: AWS_ACCESS_KEY_ID
Value: [Your AWS Access Key]

Name: AWS_SECRET_ACCESS_KEY  
Value: [Your AWS Secret Key]

Name: CLOUD_BUCKET
Value: [Your S3 bucket name, e.g., trading-bot-ml-kevin]
```

### **How to get AWS credentials:**
>>>>>>> origin/main

1. **Log into AWS Console**: https://aws.amazon.com/console/
2. **Go to IAM**: Search for "IAM" in the top search bar
3. **Create User**: 
   - Click "Users" → "Add users"
   - Username: `trading-bot-ml`
   - Access type: ✅ **Programmatic access**
4. **Set Permissions**:
   - Attach policy: `AmazonS3FullAccess`
   - (Or create custom policy with S3 access to your bucket only)
5. **Get Keys**:
   - Copy **Access Key ID** 
   - Copy **Secret Access Key**
   - ⚠️ **Save these immediately - you can't see the secret again!**

<<<<<<< HEAD
### Test Your Setup:
=======
### **Test Your Setup:**
>>>>>>> origin/main

1. **Push your code**: 
   ```bash
   git add .
   git commit -m "🌥️ Add cloud learning pipeline"
   git push
   ```

2. **Trigger manual training**:
   - Go to: `Actions` tab in your GitHub repo
<<<<<<< HEAD
   - Click: `24/7 Continuous ML/RL Training`
   - Click: `Run workflow` → `Run workflow`

### Security Notes
- Never commit AWS credentials to code
- Use IAM least-privilege principle  
- Monitor S3 costs and set up billing alerts
- Rotate HMAC key periodically

### Pipeline Features
The pipeline will run every 30 minutes and automatically:
1. Download training data from S3
2. Train 3 ML models (meta classifier, execution predictor, RL sizer)  
3. Convert to ONNX format for fast inference
4. Upload to versioned S3 paths
5. Generate and sign secure manifest with HMAC
6. Publish to CDN for bot consumption via ModelUpdaterService

=======
   - Click: `Cloud ML Training Pipeline`
   - Click: `Run workflow` → `Run workflow`

>>>>>>> origin/main
3. **Check logs**:
   - Watch the workflow run in real-time
   - Should see: "🎯 Cloud training completed successfully!"

### **🎯 Success Indicators:**

✅ **Bot uploads training data** to S3  
✅ **GitHub Actions runs** every 6 hours  
✅ **New models appear** in S3 bucket  
✅ **Bot downloads** improved models  
✅ **Learning continues** 24/7 even when PC is off!

---

**Need help?** The setup script handles most of this automatically! 🚀
