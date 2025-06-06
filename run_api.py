#!/usr/bin/env python3
"""Run the SDEUL REST API server."""

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "sdeul.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )