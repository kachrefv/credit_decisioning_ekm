import uvicorn
import argparse
import sys
import os

# Add src to path just in case not installed
sys.path.append(os.path.join(os.getcwd(), "src"))

def main():
    parser = argparse.ArgumentParser(description="Credit Decision Memory API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    # In a real enterprise app, we'd use dotenv.load_dotenv() here
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run("ekm.api.app:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()