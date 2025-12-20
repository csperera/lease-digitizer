"""
Lease Librarian API Server
Single process, no auto-reload, no zombies!
"""
import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    print("=" * 70)
    print("LEASE LIBRARIAN API SERVER")
    print("=" * 70)
    print()
    print("Starting API server on PORT 8001...")
    print("API will be available at: http://localhost:8001")
    print("API docs at: http://localhost:8001/docs")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8001,  # Using 8001 to avoid zombie conflicts
        reload=False,  # CRITICAL: Prevents zombie child processes on Windows
        log_level="info"
    )