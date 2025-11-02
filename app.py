from flask import Flask, render_template, redirect, url_for, session
from extensions import db, bcrypt
from routes.auth import auth_bp
from routes.upload import upload_bp
import os

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
import os
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.expanduser("~/dashboard.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Initialize extensions
db.init_app(app)
bcrypt.init_app(app)

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(upload_bp)

@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('dashboard.html')
    return redirect(url_for('auth.login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
