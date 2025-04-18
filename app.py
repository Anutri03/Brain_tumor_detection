import os
from datetime import datetime
import uuid
import cv2
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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

# Load the trained model
try:
    model = load_model('efficientnetB0.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def preprocess_image(img_path):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(150, 150))  # Change back to 150x150 as per training
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Simple normalization as used during training
        img_array = img_array / 255.0
        
        # Print debug information
        print(f"Preprocessed image shape: {img_array.shape}")
        print(f"Preprocessed image min/max values: {img_array.min()}/{img_array.max()}")
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise

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
                
                # Preprocess the image
                processed_image = preprocess_image(filepath)
                
                # Make prediction
                prediction = model.predict(processed_image)
                print(f"Raw prediction output: {prediction}")  # Debug print
                
                # Get the predicted class
                class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
                
                # Get the class with highest probability
                predicted_class_index = np.argmax(prediction[0])
                predicted_class = class_names[predicted_class_index]
                
                # Get probabilities directly from model output
                probabilities = prediction[0]
                confidence = float(probabilities[predicted_class_index])
                
                # Create probability dictionary
                class_probabilities = {}
                for i, class_name in enumerate(class_names):
                    prob = float(probabilities[i])
                    print(f"Probability for {class_name}: {prob}")  # Debug print
                    class_probabilities[class_name] = prob
                
                # Format the results
                result = {
                    'prediction': predicted_class.title(),
                    'confidence': f'{confidence:.2%}',
                    'image_path': os.path.join('uploads', unique_filename),
                    'class_probabilities': class_probabilities
                }
                
                print(f"Final result object: {result}")  # Debug print
                
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

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create processed image with tumor highlight
    processed_img = img.copy()
    if contours:
        # Get the largest contour (assumed to be the tumor)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(processed_img, [largest_contour], -1, (0, 255, 0), 2)
    
    # Create segmentation mask
    segmentation = np.zeros_like(gray)
    if contours:
        cv2.drawContours(segmentation, [largest_contour], -1, 255, -1)
    
    # Save processed images
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed.jpg')
    segmentation_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmentation.jpg')
    
    cv2.imwrite(processed_path, processed_img)
    cv2.imwrite(segmentation_path, segmentation)
    
    return processed_path, segmentation_path

def generate_report_data(image_path, processed_path, segmentation_path):
    # This is a placeholder for actual tumor detection logic
    # In a real application, you would use your trained model here
    tumor_detected = True
    tumor_type = "Glioma"
    confidence = 0.85
    
    return {
        'primary_findings': f"Tumor detected with {confidence*100:.1f}% confidence",
        'tumor_characteristics': f"Type: {tumor_type}\nSize: 2.5cm x 1.8cm\nLocation: Left frontal lobe",
        'recommendations': [
            "Schedule a follow-up consultation with a neurosurgeon",
            "Consider additional imaging tests for detailed analysis",
            "Monitor symptoms and report any changes",
            "Discuss treatment options with your healthcare provider"
        ],
        'original_image': image_path,
        'processed_image': processed_path,
        'segmentation_image': segmentation_path
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(filepath)
        
        # Process the image
        processed_path, segmentation_path = process_image(filepath)
        
        # Generate report data
        report_data = generate_report_data(filepath, processed_path, segmentation_path)
        
        # Generate report ID and patient ID
        report_id = str(uuid.uuid4())
        patient_id = str(uuid.uuid4())[:8]
        
        # Store report data in session or database
        # For this example, we'll just return the data
        return jsonify({
            'report_id': report_id,
            'patient_id': patient_id,
            'success': True
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/report')
def report():
    report_id = request.args.get('report_id')
    patient_id = request.args.get('patient_id')
    
    # In a real application, you would fetch this data from a database
    # For this example, we'll use placeholder data
    report_data = {
        'current_date': datetime.now().strftime('%Y-%m-%d'),
        'patient_id': patient_id,
        'report_id': report_id,
        'primary_findings': "Tumor detected with 85% confidence",
        'tumor_characteristics': "Type: Glioma\nSize: 2.5cm x 1.8cm\nLocation: Left frontal lobe",
        'recommendations': [
            "Schedule a follow-up consultation with a neurosurgeon",
            "Consider additional imaging tests for detailed analysis",
            "Monitor symptoms and report any changes",
            "Discuss treatment options with your healthcare provider"
        ],
        'original_image': '/static/uploads/original.jpg',
        'processed_image': '/static/uploads/processed.jpg',
        'segmentation_image': '/static/uploads/segmentation.jpg'
    }
    
    return render_template('report.html', **report_data)

# Ensure CSRF token is available in all templates
@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf())

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 