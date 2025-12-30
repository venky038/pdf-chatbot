/**
 * Authentication Management Script
 * Handles user login and registration flows via the QueryMate API.
 */

const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");
const errorMessage = document.getElementById("error-message");
const registerMessage = document.getElementById("register-message");

const API_BASE = "http://127.0.0.1:8000";

/**
 * --- HANDLE LOGIN ---
 * Submits credentials to exchange for a JWT access token.
 * Note: Uses FormData as required by FastAPI's OAuth2PasswordBearer.
 */
loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorMessage.textContent = "";

    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;

    const formData = new FormData();
    formData.append("username", username);
    formData.append("password", password);

    try {
        const res = await fetch(`${API_BASE}/login`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.detail || "Authentication failed. Please check your credentials.");
        }

        const data = await res.json();

        // Persist session: Save the JWT token in local storage
        localStorage.setItem("accessToken", data.access_token);

        // Navigate to the main application workspace
        window.location.href = "index.html";

    } catch (error) {
        errorMessage.textContent = error.message;
    }
});

/**
 * --- HANDLE REGISTRATION ---
 * Creates a new user identity in the database.
 */
registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    registerMessage.textContent = "";

    const username = document.getElementById("reg-username").value;
    const password = document.getElementById("reg-password").value;

    try {
        const res = await fetch(`${API_BASE}/register`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password }),
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || "Could not create account.");
        }

        // Inform user and reset form for login
        registerMessage.textContent = "Account created! You can now sign in.";
        registerMessage.classList.add("success");
        registerForm.reset();

    } catch (error) {
        registerMessage.textContent = error.message;
        registerMessage.classList.add("error");
    }
});