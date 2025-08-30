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

### **Test Your Setup:**

1. **Push your code**: 
   ```bash
   git add .
   git commit -m "🌥️ Add cloud learning pipeline"
   git push
   ```

2. **Trigger manual training**:
   - Go to: `Actions` tab in your GitHub repo
   - Click: `Cloud ML Training Pipeline`
   - Click: `Run workflow` → `Run workflow`

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
