TITLE: Currency Note Authentication using OpenCV & Machine Learning

Overview:

This project aims to detect whether a currency note is **real or fake** by analyzing its image using a **machine learning model** trained on statistical features. Instead of relying on watermark detection or complex hardware, 
we use an image of the note and apply feature extraction techniques to let the model make a smart decision.
CODE DESCRIPTION:
Python
OpenCV 
Scikit-learn

Main Use Case:

Imagine someone is using a camera (like on a mobile device or ATM scanner) to verify a currency note in real-time. 
Instead of expensive hardware or complex watermark readers, this approach allows for a **software-only lightweight solution.

Target Use Cases:
* Retail point-of-sale systems
* ATM currency validation
* Mobile banking apps
* Educational projects and demonstrations

 WORKING:

1. Model Training
   A dataset of pre-extracted statistical features is used to train a `RandomForestClassifier` with high accuracy.

2. User Uploads Image
   In the testing phase, the user uploads a photo of a currency note in Google Colab.

3. Feature Extraction
   The image is processed using OpenCV and NumPy to extract:
   * Variance
   * Skewness
   * Curtosis
   * Entropy

4. Model Prediction
   The extracted features are passed to the trained model, which returns a prediction:
   * `1` → Real Note
   * `0` → Fake Note

5. Result Display
   The result is printed to the console, letting the user know if the note is real or counterfeit.

Future Enhancements:

This is just a starting point. Here are some exciting directions this project can evolve into:

Image Preprocessing Improvements: Enhance with edge detection, noise filtering, or texture analysis for better accuracy.
Deep Learning: Replace the feature-based model with a CNN to directly classify images.
Mobile Integration: Build an Android/iOS app using this backend for on-the-go currency verification.
Multi-currency Support: Extend to support different types of currencies and denominations.
Dataset Expansion: Collect real-world note images for training a more robust model.

Conclusion:

This project demonstrates how even basic machine learning models combined with image processing can solve real-world problems like currency authentication.
It’s a lightweight, explainable, and practical approach that proves that AI doesn’t always need to be complex to be effective
Perfect for students, educators, or developers looking for a computer vision + ML mini project with real-world relevance.

