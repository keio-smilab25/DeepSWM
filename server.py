#!/usr/bin/env python3
import http.server
import socketserver
import os
import mimetypes
import socket
from urllib.parse import urlparse

class SPAHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Parse the URL
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        print(f"Requested path: {path}")
        
        # Handle /forecast route specifically - serve forecast app directly
        if path == '/forecast' or path == '/forecast/':
            try:
                forecast_file = os.path.join(os.getcwd(), 'web', 'forecast', 'index.html')
                if os.path.isfile(forecast_file):
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    with open(forecast_file, 'rb') as f:
                        self.wfile.write(f.read())
                    return
            except Exception as e:
                print(f"Error serving forecast: {e}")
                self.send_error(500)
                return
        
        # Handle legacy /demo route - redirect to /forecast
        if path == '/demo' or path == '/demo/':
            self.send_response(301)
            self.send_header('Location', '/forecast/')
            self.end_headers()
            return
        
        # If the path exists as a file, serve it normally
        if os.path.isfile(self.translate_path(path)):
            super().do_GET()
            return
            
        # If it's a directory and has an index.html, serve that
        if os.path.isdir(self.translate_path(path)):
            index_path = os.path.join(self.translate_path(path), 'index.html')
            if os.path.isfile(index_path):
                super().do_GET()
                return
        
        # For SPA routes, serve index.html
        if not path.startswith('/web/') and not '.' in os.path.basename(path):
            self.path = '/index.html'
            super().do_GET()
            return
            
        # Default behavior for other cases
        super().do_GET()

def find_free_port(start_port=8081, max_port=8100):
    """利用可能なポートを見つける"""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start_port}-{max_port}")

if __name__ == "__main__":
    try:
        PORT = find_free_port()
        
        with socketserver.TCPServer(("0.0.0.0", PORT), SPAHTTPRequestHandler) as httpd:
            print("=" * 60)
            print("🚀 Deep Space Weather Model - Development Server")
            print("=" * 60)
            print(f"🌐 Server running at: http://localhost:{PORT}")
            print(f"🚀 Forecast page: http://localhost:{PORT}/forecast")
            print(f"🔬 Demo redirect: http://localhost:{PORT}/demo → /forecast")
            print("=" * 60)
            print("📝 GitHub Pages Deployment Notes:")
            print("   • GitHub Pages serves static files only")
            print("   • This server.py is for local development only")
            print("   • For GitHub Pages, use client-side routing")
            print("   • See deployment instructions below")
            print("=" * 60)
            print("Press Ctrl+C to stop the server")
            print()
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 Server stopped.")
                
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("💡 Try stopping other servers or use a different port range")
