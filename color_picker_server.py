"""
Flask server for advanced HTML/JS color picker
Communicates with gestureOS.py via REST API
"""

from flask import Flask, render_template, request, jsonify
import json
from pathlib import Path
import threading
import time

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global color state (BGR format for OpenCV)
current_color = {
    'hex': '#00FF00',  # Green
    'rgb': (0, 255, 0),
    'bgr': (0, 255, 0),  # OpenCV uses BGR
    'hsv': (60, 100, 100),
    'updated': False
}

color_lock = threading.Lock()


def hex_to_bgr(hex_color):
    """Convert hex color (#RRGGBB) to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # BGR format


def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    
    # Hue
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    else:
        h = (60 * ((r - g) / df) + 240) % 360
    
    # Saturation
    s = 0 if mx == 0 else (df / mx) * 100
    
    # Value
    v = mx * 100
    
    return (int(h), int(s), int(v))


@app.route('/')
def index():
    """Serve the color picker page."""
    return render_template('color_picker.html')


@app.route('/api/color', methods=['GET', 'POST'])
def color_api():
    """
    GET: Return current color
    POST: Update color from picker
    """
    global current_color
    
    if request.method == 'POST':
        data = request.get_json()
        hex_color = data.get('hex', '#00FF00')
        
        with color_lock:
            current_color['hex'] = hex_color
            
            # Convert hex to RGB
            r = int(hex_color.lstrip('#')[0:2], 16)
            g = int(hex_color.lstrip('#')[2:4], 16)
            b = int(hex_color.lstrip('#')[4:6], 16)
            
            current_color['rgb'] = (r, g, b)
            current_color['bgr'] = hex_to_bgr(hex_color)  # For OpenCV
            current_color['hsv'] = rgb_to_hsv(r, g, b)
            current_color['updated'] = True
        
        return jsonify({
            'success': True,
            'color': {
                'hex': hex_color,
                'rgb': current_color['rgb'],
                'bgr': current_color['bgr'],
                'hsv': current_color['hsv']
            }
        })
    
    else:  # GET
        with color_lock:
            return jsonify({
                'hex': current_color['hex'],
                'rgb': current_color['rgb'],
                'bgr': current_color['bgr'],
                'hsv': current_color['hsv'],
                'updated': current_color['updated']
            })


@app.route('/api/color/bgr', methods=['GET'])
def get_color_bgr():
    """Get current color in BGR format for OpenCV."""
    with color_lock:
        return jsonify({
            'bgr': current_color['bgr'],
            'hex': current_color['hex']
        })


def run_server(port=5000):
    """Run Flask server in a separate thread."""
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    print(f"🎨 Color Picker Server starting...")
    print(f"📍 Open: http://127.0.0.1:5000")
    run_server()
