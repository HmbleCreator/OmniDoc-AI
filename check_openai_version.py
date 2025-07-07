#!/usr/bin/env python3
"""
Check OpenAI version and provide installation instructions
"""

import sys

def check_openai_version():
    """Check OpenAI version and compatibility"""
    print("🔍 Checking OpenAI Installation")
    print("=" * 40)
    
    try:
        import openai
        print(f"✅ OpenAI package found")
        print(f"   - Version: {openai.__version__}")
        
        # Check if we have the new OpenAI client
        if hasattr(openai, 'OpenAI'):
            print("✅ OpenAI v1.0+ detected - compatible with OmniDoc AI")
            return True
        else:
            print("❌ OpenAI version too old - needs v1.0+")
            print("   - Current version doesn't have OpenAI() client")
            return False
            
    except ImportError:
        print("❌ OpenAI package not installed")
        return False

def provide_installation_instructions():
    """Provide installation instructions"""
    print("\n📦 Installation Instructions")
    print("=" * 40)
    
    print("To fix OpenAI compatibility, run one of these commands:")
    print()
    print("Option 1 - Upgrade existing installation:")
    print("   pip install --upgrade openai")
    print()
    print("Option 2 - Install fresh (recommended):")
    print("   pip uninstall openai")
    print("   pip install openai>=1.0.0")
    print()
    print("Option 3 - Install with specific version:")
    print("   pip install openai==1.3.0")
    print()
    print("After installation, restart your Python environment.")

def test_openai_client():
    """Test OpenAI client functionality"""
    print("\n🧪 Testing OpenAI Client")
    print("=" * 40)
    
    try:
        import openai
        
        if hasattr(openai, 'OpenAI'):
            # Test client creation (without API key)
            client = openai.OpenAI(api_key="test-key")
            print("✅ OpenAI client can be created")
            
            # Check available methods
            if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
                print("✅ OpenAI client has chat.completions method")
                return True
            else:
                print("❌ OpenAI client missing expected methods")
                return False
        else:
            print("❌ OpenAI client not available")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OpenAI client: {e}")
        return False

def main():
    """Main function"""
    print("🚀 OmniDoc AI - OpenAI Version Check")
    print("=" * 50)
    
    # Check version
    is_compatible = check_openai_version()
    
    if not is_compatible:
        provide_installation_instructions()
    else:
        # Test client functionality
        test_openai_client()
        print("\n✅ OpenAI is ready to use with OmniDoc AI!")
    
    print("\n" + "=" * 50)
    print("💡 Tip: After fixing OpenAI, restart your backend server")

if __name__ == "__main__":
    main() 