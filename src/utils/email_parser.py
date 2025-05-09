"""
Email Parser Module for Phishing Detection

This module handles the parsing of email content to extract headers, body, and URLs.
"""

import re
import email
from email.header import decode_header
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin


def parse_email(email_content):
    """Parse an email string into its components.
    
    Args:
        email_content (str): Raw email content
        
    Returns:
        dict: Email components (headers, body, URLs, etc.)
    """
    if not email_content or not isinstance(email_content, str):
        raise ValueError("Email content must be a non-empty string")
    
    # Parse email
    try:
        msg = email.message_from_string(email_content)
    except Exception as e:
        # If parsing fails, try a more lenient approach
        # This handles cases where the email content is just the body without headers
        return {
            'headers': {},
            'subject': "",
            'from': "",
            'to': "",
            'date': "",
            'body': email_content,
            'urls': extract_urls(email_content),
            'has_attachments': False,
            'is_html': '<html' in email_content.lower()
        }
    
    # Extract headers
    headers = {}
    for key in msg.keys():
        headers[key] = decode_email_header(msg[key])
    
    # Extract common header fields
    subject = decode_email_header(msg.get('Subject', ''))
    from_addr = decode_email_header(msg.get('From', ''))
    to_addr = decode_email_header(msg.get('To', ''))
    date = decode_email_header(msg.get('Date', ''))
    
    # Extract body
    body = extract_email_body(msg)
    
    # Extract URLs from body
    urls = extract_urls(body)
    
    # Check for attachments
    has_attachments = any(part.get_filename() for part in msg.walk() if part.get_filename())
    
    # Determine if body is HTML
    is_html = '<html' in body.lower()
    
    # Return parsed email
    return {
        'headers': headers,
        'subject': subject,
        'from': from_addr,
        'to': to_addr,
        'date': date,
        'body': body,
        'urls': urls,
        'has_attachments': has_attachments,
        'is_html': is_html
    }


def decode_email_header(header_value):
    """Decode email header value.
    
    Args:
        header_value (str): Encoded header value
        
    Returns:
        str: Decoded header value
    """
    if not header_value:
        return ""
    
    try:
        decoded_parts = []
        for part, encoding in decode_header(header_value):
            if isinstance(part, bytes):
                if encoding:
                    try:
                        decoded_parts.append(part.decode(encoding))
                    except (UnicodeDecodeError, LookupError):
                        # Fallback to utf-8 if specified encoding fails
                        try:
                            decoded_parts.append(part.decode('utf-8', errors='replace'))
                        except:
                            decoded_parts.append(part.decode('ascii', errors='replace'))
                else:
                    # No encoding specified, try utf-8
                    try:
                        decoded_parts.append(part.decode('utf-8', errors='replace'))
                    except:
                        decoded_parts.append(part.decode('ascii', errors='replace'))
            else:
                decoded_parts.append(part)
                
        return " ".join(decoded_parts)
    except Exception as e:
        # If decoding fails, return as is
        return header_value


def extract_email_body(msg):
    """Extract body content from an email message.
    
    Args:
        msg (email.message.Message): Email message
        
    Returns:
        str: Email body content
    """
    body = ""
    
    # If the message is multipart
    if msg.is_multipart():
        # Prefer HTML parts over plain text
        html_parts = []
        text_parts = []
        
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
                
            if content_type == "text/html":
                try:
                    html_parts.append(part.get_payload(decode=True).decode('utf-8', errors='replace'))
                except:
                    try:
                        html_parts.append(part.get_payload(decode=True).decode('latin-1', errors='replace'))
                    except:
                        pass
            elif content_type == "text/plain":
                try:
                    text_parts.append(part.get_payload(decode=True).decode('utf-8', errors='replace'))
                except:
                    try:
                        text_parts.append(part.get_payload(decode=True).decode('latin-1', errors='replace'))
                    except:
                        pass
        
        # Prefer HTML content if available
        if html_parts:
            body = '\n'.join(html_parts)
        elif text_parts:
            body = '\n'.join(text_parts)
    else:
        # If the message is not multipart
        content_type = msg.get_content_type()
        
        try:
            body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
        except:
            try:
                body = msg.get_payload(decode=True).decode('latin-1', errors='replace')
            except:
                body = msg.get_payload()
                if isinstance(body, bytes):
                    body = body.decode('ascii', errors='replace')
    
    return body


def extract_urls(text):
    """Extract URLs from text.
    
    Args:
        text (str): Text to extract URLs from
        
    Returns:
        list: List of URLs
    """
    if not text or not isinstance(text, str):
        return []
    
    urls = []
    
    # Handle text content
    # URL pattern for plain text
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+\.[^\s<>"]+'
    text_urls = re.findall(url_pattern, text)
    urls.extend(text_urls)
    
    # Handle HTML content
    if '<html' in text.lower():
        try:
            soup = BeautifulSoup(text, 'html.parser')
            
            # Extract URLs from href attributes
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href'].strip()
                if href and not href.startswith('#') and not href.startswith('mailto:'):
                    # Handle relative URLs
                    if not href.startswith(('http://', 'https://', 'www.')):
                        # Without a base URL, we can't resolve relative URLs properly
                        # Just add them as is
                        urls.append(href)
                    else:
                        urls.append(href)
            
            # Extract URLs from form actions
            for form in soup.find_all('form', action=True):
                action = form['action'].strip()
                if action and not action.startswith('#'):
                    # Handle relative URLs
                    if not action.startswith(('http://', 'https://', 'www.')):
                        urls.append(action)
                    else:
                        urls.append(action)
            
            # Extract URLs from images and scripts
            for tag in soup.find_all(['img', 'script'], src=True):
                src = tag['src'].strip()
                if src and not src.startswith('data:'):
                    if not src.startswith(('http://', 'https://', 'www.')):
                        urls.append(src)
                    else:
                        urls.append(src)
        except Exception as e:
            # If HTML parsing fails, continue with what we have
            pass
    
    # Remove duplicates while preserving order
    unique_urls = []
    for url in urls:
        if url not in unique_urls:
            unique_urls.append(url)
    
    return unique_urls


def extract_email_metadata(parsed_email):
    """Extract metadata features from a parsed email.
    
    Args:
        parsed_email (dict): Parsed email from parse_email()
        
    Returns:
        dict: Metadata features
    """
    headers = parsed_email['headers']
    
    # Initialize metadata features
    metadata = {
        'has_reply_to': 'Reply-To' in headers,
        'has_return_path': 'Return-Path' in headers,
        'has_authentication_results': 'Authentication-Results' in headers,
        'has_dkim_signature': 'DKIM-Signature' in headers,
        'has_spf': False,
        'spf_result': 'none',
        'has_dkim': False,
        'dkim_result': 'none',
        'priority_level': 'normal',
        'num_recipients': 1,
        'sent_to_undisclosed_recipients': False,
        'from_contains_name': False,
        'reply_to_different_from_from': False
    }
    
    # Extract SPF and DKIM results
    if 'Authentication-Results' in headers:
        auth_results = headers['Authentication-Results']
        
        # Check SPF
        if 'spf=' in auth_results:
            metadata['has_spf'] = True
            spf_match = re.search(r'spf=(\w+)', auth_results)
            if spf_match:
                metadata['spf_result'] = spf_match.group(1)
        
        # Check DKIM
        if 'dkim=' in auth_results:
            metadata['has_dkim'] = True
            dkim_match = re.search(r'dkim=(\w+)', auth_results)
            if dkim_match:
                metadata['dkim_result'] = dkim_match.group(1)
    
    # Check priority
    if 'X-Priority' in headers:
        priority = headers['X-Priority']
        if priority in ['1', '2']:
            metadata['priority_level'] = 'high'
        elif priority in ['4', '5']:
            metadata['priority_level'] = 'low'
    elif 'Importance' in headers:
        importance = headers['Importance'].lower()
        if importance == 'high':
            metadata['priority_level'] = 'high'
        elif importance == 'low':
            metadata['priority_level'] = 'low'
    
    # Count recipients
    if parsed_email['to']:
        to_field = parsed_email['to']
        if 'undisclosed-recipients' in to_field:
            metadata['sent_to_undisclosed_recipients'] = True
        else:
            # Count commas to estimate number of recipients
            metadata['num_recipients'] = to_field.count(',') + 1
    
    # Check if From field contains a name
    from_field = parsed_email['from']
    if from_field:
        metadata['from_contains_name'] = '<' in from_field and '>' in from_field
    
    # Check if Reply-To is different from From
    if 'Reply-To' in headers and from_field:
        reply_to = headers['Reply-To']
        # Extract email address from From field if it contains a name
        from_email = re.search(r'<([^>]+)>', from_field)
        if from_email:
            from_email = from_email.group(1)
        else:
            from_email = from_field
        
        # Extract email address from Reply-To field if it contains a name
        reply_to_email = re.search(r'<([^>]+)>', reply_to)
        if reply_to_email:
            reply_to_email = reply_to_email.group(1)
        else:
            reply_to_email = reply_to
        
        metadata['reply_to_different_from_from'] = from_email.lower() != reply_to_email.lower()
    
    return metadata


if __name__ == "__main__":
    # Example usage
    sample_email = """From: John Doe <john@example.com>
To: Jane Smith <jane@example.com>
Subject: Important Security Update
Date: Wed, 1 Mar 2023 12:00:00 -0500
Content-Type: text/html; charset="utf-8"

<html>
<body>
<p>Dear Customer,</p>
<p>Your account security needs to be updated. Please click this link: <a href="https://malicious-site.com/update">Update Now</a></p>
<p>Regards,<br>Security Team</p>
</body>
</html>
"""
    
    # Parse email
    parsed_email = parse_email(sample_email)
    
    # Print parsed email
    for key, value in parsed_email.items():
        if key == 'headers':
            print(f"{key}:")
            for header_key, header_value in value.items():
                print(f"  {header_key}: {header_value}")
        elif key == 'urls':
            print(f"{key}:")
            for url in value:
                print(f"  {url}")
        else:
            print(f"{key}: {value}")
    
    # Extract metadata
    metadata = extract_email_metadata(parsed_email)
    
    # Print metadata
    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")