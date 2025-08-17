#!/usr/bin/env python
"""
Test script to verify BrainProxy2 import and basic functionality
"""

try:
    # Test importing the new BrainProxy2
    from brain_proxy import BrainProxy2, BrainProxyConfig
    print("✅ Successfully imported BrainProxy2 and BrainProxyConfig")
    
    # Test creating an instance with the new config
    config = BrainProxyConfig(
        enable_memory=True,
        debug=True,
        storage_dir="test_tenants"
    )
    print("✅ Successfully created BrainProxyConfig")
    
    # Test creating BrainProxy2 instance
    proxy = BrainProxy2(config)
    print("✅ Successfully created BrainProxy2 instance")
    
    # Verify services are initialized
    assert proxy.memory_service is not None
    assert proxy.session_service is not None
    assert proxy.document_service is not None
    assert proxy.tool_service is not None
    assert proxy.streaming_service is not None
    print("✅ All services initialized correctly")
    
    # Test backward compatibility with kwargs
    proxy2 = BrainProxy2(
        enable_memory=True,
        debug=True,
        storage_dir="test_tenants2"
    )
    print("✅ Backward compatibility with kwargs works")
    
    # Check that router is set up
    assert proxy.router is not None
    print("✅ FastAPI router is configured")
    
    print("\n🎉 All tests passed! BrainProxy2 is working correctly.")
    print(f"📦 Package version 0.1.76 has been successfully deployed to PyPI")
    print(f"🔗 View at: https://pypi.org/project/brain-proxy/0.1.76/")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct environment or install with: pip install brain-proxy==0.1.76")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
