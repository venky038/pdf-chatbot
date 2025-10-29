const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");
const errorMessage = document.getElementById("error-message");
const registerMessage = document.getElementById("register-message");

const API_BASE = "http://127.0.0.1:8000";

// --- Handle Login ---
loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorMessage.textContent = "";
    
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;

    // FastAPI's OAuth2 expects form-data, not JSON
    const formData = new FormData();
    formData.append("username", username);
    formData.append("password", password);

    try {
        const res = await fetch(`${API_BASE}/login`, {
            method: "POST",
            body: formData, // Send as form data
        });

        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.detail || "Login failed");
        }

        const data = await res.json();
        
        // Save the token and redirect
        localStorage.setItem("accessToken", data.access_token);
        window.location.href = "index.html"; // Redirect to the main chat app

    } catch (error) {
        errorMessage.textContent = error.message;
    }
});

// --- Handle Registration ---
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
            throw new Error(data.detail || "Registration failed");
        }

        registerMessage.textContent = data.message;
        registerMessage.className = "success";
        registerForm.reset();

    } catch (error) {
        registerMessage.textContent = error.message;
        registerMessage.className = "error";
    }
});