import os
from datetime import datetime
import uuid
import cv2
import numpy as np
from PIL import Image
import io

# Flask and extensions
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect, generate_csrf

# Werkzeug utilities
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Database and security
from sqlalchemy.exc import IntegrityError
from functools import wraps

# Import prediction functions
from predict import load_trained_model, predict_tumor

app = Flask(__name__)
# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Replace random key with a fixed one
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')  # Update upload folder path
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['REMEMBER_COOKIE_DURATION'] = 60 * 60 * 24 * 30  # 30 days
app.config['SESSION_PROTECTION'] = 'strong'
app.config['WTF_CSRF_ENABLED'] = True
app.config['WTF_CSRF_SECRET_KEY'] = 'csrf-secret-key'  # Add a fixed CSRF secret key

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
csrf = CSRFProtect(app)
csrf.init_app(app)  # Ensure CSRF protection is properly initialized
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'signin'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Load the model at startup
model = load_trained_model()

ALLOWED_EXTENSIONS = {
    'dcm', 'nii', 'jpg', 'jpeg', 'png', 'tiff', 'bmp', 'nrrd', 'gz', 'pdf'
}

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    login_attempts = db.Column(db.Integer, default=0, nullable=False)
    scans = db.relationship('ScanHistory', backref='user', lazy=True)
    
    def __init__(self, fullname, email, password, is_admin=False):
        self.fullname = fullname
        self.email = email.lower()
        self.password = generate_password_hash(password, method='pbkdf2:sha256:260000')
        self.is_admin = is_admin
        self.login_attempts = 0
        self.is_active = True

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def update_last_login(self):
        self.last_login = datetime.utcnow()
        self.login_attempts = 0
        db.session.commit()

    def increment_login_attempts(self):
        if self.login_attempts is None:
            self.login_attempts = 1
        else:
            self.login_attempts += 1
        db.session.commit()

    @property
    def is_locked(self):
        return self.login_attempts is not None and self.login_attempts >= 5

# Session Model for tracking user sessions
class UserSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)

# Scan History Model
class ScanHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    research_link = db.Column(db.String(255))

    def __init__(self, user_id, image_path, result, confidence, type=None, research_link=None):
        self.user_id = user_id
        self.image_path = image_path
        self.result = result
        self.confidence = confidence
        self.type = type
        self.research_link = research_link

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Landing page - accessible without login
@app.route('/')
def home():
    return render_template('landingpagemain.html')

# Authentication routes
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        password = request.form.get('password')
        remember = bool(request.form.get('remember'))

        if not email or not password:
            flash('Please enter both email and password.', 'error')
            return redirect(url_for('signin'))

        user = User.query.filter_by(email=email).first()

        if not user:
            flash('Invalid email or password.', 'error')
            return redirect(url_for('signin'))

        if user.is_locked:
            flash('Account is locked due to too many failed attempts. Please contact support.', 'error')
            return redirect(url_for('signin'))

        if not user.check_password(password):
            user.increment_login_attempts()
            remaining_attempts = 5 - user.login_attempts
            if remaining_attempts > 0:
                flash(f'Invalid password. {remaining_attempts} attempts remaining.', 'error')
            else:
                flash('Account is now locked due to too many failed attempts. Please contact support.', 'error')
            return redirect(url_for('signin'))

        if not user.is_active:
            flash('This account has been deactivated.', 'error')
            return redirect(url_for('signin'))

        # Successful login
        user.update_last_login()
        login_user(user, remember=remember)
        flash('Successfully logged in!', 'success')
        
        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('home')
        return redirect(next_page)

    return render_template('signin.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        try:
            email = request.form.get('email', '').lower()
            fullname = request.form.get('fullname')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm-password')

            # Validation
            if not all([email, fullname, password, confirm_password]):
                flash('All fields are required.', 'error')
                return redirect(url_for('register'))

            if password != confirm_password:
                flash('Passwords do not match!', 'error')
                return redirect(url_for('register'))

            if len(password) < 8:
                flash('Password must be at least 8 characters long.', 'error')
                return redirect(url_for('register'))

            # Create new user
            new_user = User(
                fullname=fullname,
                email=email,
                password=password
            )
            db.session.add(new_user)
            db.session.commit()

            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('signin'))

        except IntegrityError:
            db.session.rollback()
            flash('Email address already exists!', 'error')
            return redirect(url_for('register'))

        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Successfully logged out.', 'success')
    return redirect(url_for('home'))

# Protected routes - require login
@app.route('/technology')
def technology():
    return render_template('technology.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/research')
def research():
    return render_template('research.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Create a unique filename
                filename = secure_filename(file.filename)
                unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Ensure the upload folder exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                # Save the file
                file.save(filepath)
                
                # Check if model is loaded
                if model is None:
                    flash('Model not loaded. Please contact administrator.', 'error')
                    return redirect(request.url)
                
                # Make prediction using our predict.py function
                result = predict_tumor(model, filepath)
                
                # Add image path to result
                result['image_path'] = os.path.join('uploads', unique_filename)
                
                # Store result in session for the results page
                session['analysis_result'] = result
                
                # Redirect to results page
                return redirect(url_for('results'))
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                print(f"Error in prediction: {str(e)}")
                return redirect(request.url)
                
        else:
            flash('Invalid file type. Please upload an image file.', 'error')
            return redirect(request.url)
            
    return render_template('predict.html')

@app.route('/results')
@login_required
def results():
    result = session.get('analysis_result')
    if not result:
        flash('No analysis results found. Please upload an image first.', 'error')
        return redirect(url_for('predict'))
    
    # Save the scan to history
    try:
        new_scan = ScanHistory(
            user_id=current_user.id,
            image_path=result['image_path'],
            result=result['prediction'],
            confidence=result['confidence'],
            type=result.get('type'),
            research_link=result.get('research_link')
        )
        db.session.add(new_scan)
        db.session.commit()
    except Exception as e:
        print(f"Error saving scan to history: {str(e)}")
        # Continue even if history save fails
    
    return render_template('result.html', result=result)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/change-password', methods=['POST'])
@login_required
def change_password():
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    if not current_user.check_password(current_password):
        flash('Current password is incorrect.', 'error')
        return redirect(url_for('profile'))

    if new_password != confirm_password:
        flash('New passwords do not match.', 'error')
        return redirect(url_for('profile'))

    if len(new_password) < 8:
        flash('New password must be at least 8 characters long.', 'error')
        return redirect(url_for('profile'))

    current_user.password = generate_password_hash(new_password, method='pbkdf2:sha256:260000')
    db.session.commit()
    flash('Password successfully updated.', 'success')
    return redirect(url_for('profile'))

@app.route('/history')
@login_required
def history():
    # Get all scans for the current user, ordered by date
    scans = ScanHistory.query.filter_by(user_id=current_user.id)\
        .order_by(ScanHistory.created_at.desc())\
        .all()
    
    return render_template('history.html', scans=scans)

@app.route('/view_scan/<int:scan_id>')
@login_required
def view_scan(scan_id):
    scan = ScanHistory.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        flash('You do not have permission to view this scan.', 'error')
        return redirect(url_for('history'))
    
    result = {
        'prediction': scan.result,
        'confidence': scan.confidence,
        'image_path': scan.image_path,
        'type': scan.type,
        'research_link': scan.research_link
    }
    
    return render_template('result.html', result=result)

@app.route('/download_report/<int:scan_id>')
@login_required
def download_report(scan_id):
    scan = ScanHistory.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        flash('You do not have permission to download this report.', 'error')
        return redirect(url_for('history'))
    
    result = {
        'prediction': scan.result,
        'confidence': scan.confidence,
        'image_path': scan.image_path,
        'type': scan.type,
        'research_link': scan.research_link
    }
    
    session['analysis_result'] = result
    return redirect(url_for('results'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# Ensure CSRF token is available in all templates
@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf())

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 