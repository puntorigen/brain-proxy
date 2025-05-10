#!/usr/bin/env python
"""
Test script to upload a file to the brain-proxy API
"""

import base64
import json
import requests
import sys

def main():
    # Check if a file path is provided
    if len(sys.argv) < 2:
        print("Usage: python upload_file.py <file_path> [tenant_name]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    tenant = sys.argv[2] if len(sys.argv) > 2 else "test_tenant"
    
    # Read the file and encode it as base64
    with open(file_path, 'rb') as f:
        file_content = f.read()
        base64_encoded = base64.b64encode(file_content).decode('utf-8')
    
    # Get the filename
    filename = file_path.split("/")[-1]
    
    # Create the request payload
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a document that contains important information. Please read it and let me know when you're ready to answer questions about it."
                    },
                    {
                        "type": "file_data",
                        "file_data": {
                            "name": filename,
                            "mime": "text/plain",
                            "data": base64_encoded
                        }
                    }
                ]
            }
        ]
    }
    
    # Send the request to the API
    url = f"http://localhost:8000/v1/{tenant}/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    print(f"Uploading file {filename} to tenant {tenant}...")
    response = requests.post(url, headers=headers, json=payload)
    
    # Print the response
    print(f"Status code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)

if __name__ == "__main__":
    main()
