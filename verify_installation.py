#!/usr/bin/env python3
"""
Quick verification script for MultiNEAs package installation
"""

import sys

def main():
    print("=" * 60)
    print("MultiNEAs Package Verification")
    print("=" * 60)
    
    try:
        import multineas as mn
        print("✓ Package imported successfully")
        
        # Check version
        print(f"✓ Version: {mn.__version__}")
        
        # Check authors
        print(f"✓ Authors: {mn.__author__}")
        
        # Check email
        print(f"✓ Contact: {mn.__email__}")
        
        # Check base class
        obj = mn.MultiNEAsBase()
        print(f"✓ Base class instantiated: {type(obj).__name__}")
        
        # Check string representation
        str_repr = str(obj)
        print(f"✓ String representation works")
        
        print("\n" + "=" * 60)
        print("All checks passed! MultiNEAs is ready to use.")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
