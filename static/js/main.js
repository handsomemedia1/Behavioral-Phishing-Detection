/**
 * Behavioral Phishing Detection - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    // Form validation
    const emailForm = document.querySelector('form');
    if (emailForm) {
        emailForm.addEventListener('submit', function(event) {
            const emailContent = document.getElementById('email').value.trim();
            if (!emailContent) {
                event.preventDefault();
                alert('Please enter email content to analyze.');
                return false;
            }
            
            // Show loading indicator
            const submitButton = this.querySelector('button[type="submit"]');
            const originalText = submitButton.innerHTML;
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
            submitButton.disabled = true;
            
            // Allow form submission
            return true;
        });
    }
    
    // Example Email Button
    const exampleButton = document.getElementById('load-example');
    if (exampleButton) {
        exampleButton.addEventListener('click', function() {
            const emailTextarea = document.getElementById('email');
            const exampleEmail = `From: Bank Security <security@bank-secure-verify.com>
To: Customer <customer@example.com>
Subject: URGENT: Your Online Banking Access Will Be Suspended
Date: Mon, 15 May 2023 09:23:45 -0500

Dear Valued Customer,

We have detected SUSPICIOUS activity on your online banking account. Your account access will be SUSPENDED within 24 HOURS unless you VERIFY your information IMMEDIATELY!

Please click on the following link to verify your account details:
https://bank-secure-verification.com/login.php?user=customer@example.com

If you fail to verify your account, your online banking access will be TERMINATED and you will need to visit a local branch with TWO forms of ID to restore access.

DO NOT DELAY! This is a time-sensitive security matter!

Sincerely,
Bank Security Team
Customer Protection Department`;
            
            emailTextarea.value = exampleEmail;
        });
    }
    
    // Copy API code example
    const copyApiButton = document.getElementById('copy-api-code');
    if (copyApiButton) {
        copyApiButton.addEventListener('click', function() {
            const codeBlock = document.getElementById('api-code-example');
            
            // Create temporary textarea element
            const textarea = document.createElement('textarea');
            textarea.value = codeBlock.textContent;
            document.body.appendChild(textarea);
            
            // Select and copy text
            textarea.select();
            document.execCommand('copy');
            
            // Clean up
            document.body.removeChild(textarea);
            
            // Change button text temporarily
            const originalText = this.innerHTML;
            this.innerHTML = 'Copied!';
            setTimeout(() => {
                this.innerHTML = originalText;
            }, 2000);
        });
    }
});