// Default credentials
const validUsername = "sasisai";
const validPassword = "123110";

// Handle the login form submission
document.getElementById("loginForm").addEventListener("submit", function (event) {
    event.preventDefault();

    // Get the values from the login form
    const username = document.getElementById("login_username").value;
    const password = document.getElementById("login_password").value;

    // Check if the entered credentials are valid
    if (username === validUsername && password === validPassword) {
        // If login is successful, redirect to index.html
        alert("Login successful! Redirecting...");
        window.location.href = "index.html";  // Redirect to the index page
    } else {
        // If login fails, show an alert message
        alert("Invalid username or password. Please try again.");
    }
});