"""
Upload Demo Leases to Lease Librarian API
Reads generated demo leases and uploads to port 8001
"""
import json
import requests
from pathlib import Path

# Configuration
DEMO_FILE = "demo_leases_batch.json"
API_URL = "http://localhost:8001/api/v1/leases/batch"
TIMEOUT = 600


def upload_demo_leases():
    """Upload demo leases to API."""
    
    print("\n" + "=" * 70)
    print("UPLOADING DEMO LEASES TO LEASE LIBRARIAN")
    print("=" * 70)
    print()
    
    # Check if demo file exists
    if not Path(DEMO_FILE).exists():
        print(f"✗ Error: {DEMO_FILE} not found!")
        print()
        print("Run this first: python generate_demo_leases.py")
        return False
    
    # Load demo leases
    print(f"Loading leases from: {DEMO_FILE}")
    with open(DEMO_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    leases = data['leases']
    print(f"✓ Loaded {len(leases)} demo leases")
    print()
    
    # Show what we're uploading
    print("Demo leases to upload:")
    for i, lease in enumerate(leases, 1):
        content_preview = lease['content'][:200].replace('\n', ' ')
        print(f"  {i}. {lease['filename']}: {len(lease['content']):,} characters")
    print()
    
    # Upload to API
    print("=" * 70)
    print("UPLOADING TO API...")
    print("=" * 70)
    print()
    
    try:
        response = requests.post(
            API_URL,
            json=data,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("=" * 70)
            print("UPLOAD RESULTS")
            print("=" * 70)
            print(f"✓ Processed: {result.get('processed', 0)} lease(s)")
            
            if result.get('failed', 0) > 0:
                print(f"✗ Failed: {result.get('failed', 0)} lease(s)")
                print()
                print("Errors:")
                for error in result.get('errors', []):
                    print(f"  - {error}")
            
            if result.get('lease_ids'):
                print()
                print(f"Created {len(result['lease_ids'])} lease IDs:")
                for lease_id in result['lease_ids']:
                    print(f"  ✓ {lease_id}")
            
            print()
            print("=" * 70)
            print("SUCCESS! DEMO PORTFOLIO READY!")
            print("=" * 70)
            print()
            print("VIEW YOUR LEASES:")
            print("http://localhost:3000/lease-digitizer-final.html")
            print()
            print("You should see:")
            print("  - Portfolio Size: 30 Leases")
            print("  - All 30 lease cards in Lease Library")
            print("  - Click any lease to see extracted details")
            print("  - Chat with Lease Librarian about portfolio")
            print()
            
            return result.get('processed', 0) > 0
            
        else:
            print(f"✗ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Connection Error!")
        print()
        print("Make sure your API is running:")
        print("  python run_api_port8001.py")
        print()
        return False
        
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        return False


def main():
    success = upload_demo_leases()
    
    if success:
        print("=" * 70)
        print("NEXT STEPS FOR SUNDAY DEMO")
        print("=" * 70)
        print()
        print("1. ✓ Test Lease Librarian chat")
        print("     Try: 'What's my total portfolio value?'")
        print("     Try: 'Show me leases expiring in the next 2 years'")
        print()
        print("2. ✓ Add PDF serving endpoint (optional)")
        print("     So 'Open PDF' buttons work")
        print()
        print("3. ✓ Take screenshots for demo")
        print()
        print("4. ✓ Practice demo flow")
        print()
        print("You're ready to impress Ash! 🚀")
        print()
    else:
        print("Upload did not complete successfully.")
        print("Check errors above and try again.")


if __name__ == "__main__":
    main()
