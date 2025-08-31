#!/bin/bash

# 🚀 Complete 24/7 Cloud Learning Setup Script
# Run this to set up everything automatically

echo "🎯 Setting up 24/7 Cloud Learning Pipeline..."

# 1. Generate HMAC key for manifest signing
HMAC_KEY=$(openssl rand -hex 32)
echo "✅ Generated HMAC key: $HMAC_KEY"

# 2. Set up sample AWS credentials (replace with your real ones)
echo ""
echo "📋 GITHUB SECRETS TO CONFIGURE:"
echo "Repository Settings > Secrets and variables > Actions > New repository secret"
echo ""
echo "AWS_ACCESS_KEY_ID:"
echo "AKIAEXAMPLEKEYID123"
echo ""
echo "AWS_SECRET_ACCESS_KEY:"
echo "ExampleSecretKey+AbCdEfGhIjKlMnOpQrStUvWxYz123"
echo ""
echo "AWS_REGION:"
echo "us-east-1"
echo ""
echo "S3_BUCKET:"
echo "kevin-trading-bot-models-$(date +%s)"
echo ""
echo "CDN_BASE_URL:"
echo "https://kevin-trading-bot-models-$(date +%s).s3.amazonaws.com"
echo ""
echo "MANIFEST_HMAC_KEY:"
echo "$HMAC_KEY"
echo ""

# 3. Create AWS CLI commands for bucket setup
echo "🪣 AWS S3 BUCKET SETUP COMMANDS:"
echo "aws s3 mb s3://kevin-trading-bot-models-$(date +%s)"
echo "aws s3api put-bucket-policy --bucket kevin-trading-bot-models-$(date +%s) --policy file://bucket-policy.json"
echo ""

# 4. Generate bucket policy
cat > bucket-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowGitHubActions",
      "Effect": "Allow",
      "Principal": {
        "AWS": "*"
      },
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::kevin-trading-bot-models-*/*"
    },
    {
      "Sid": "AllowPublicRead",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::kevin-trading-bot-models-*/models/*"
    }
  ]
}
EOF

echo "✅ Created bucket-policy.json"
echo ""
echo "🎉 Setup Complete! Next steps:"
echo "1. Go to GitHub repository settings"
echo "2. Add the secrets listed above"  
echo "3. Run the AWS commands to create your S3 bucket"
echo "4. Your AI will start learning every 30 minutes automatically!"
echo ""
echo "📊 Monitor progress at: https://github.com/kevinsuero072897-collab/trading-bot-c-/actions"
