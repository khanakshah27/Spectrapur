// Spectrapur frontend configuration.
//
// After you deploy the backend (Render), replace the URL below with your
// live backend URL, e.g. "https://spectrapur-api.onrender.com".
// While developing locally with `python app.py`, the default below is fine.
window.SPECTRAPUR_CONFIG = {
  API_BASE_URL: (function () {
    const host = window.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") {
      return "http://localhost:5000";
    }
    // TODO: set this to your deployed Render backend URL.
    return "https://YOUR-RENDER-BACKEND-URL.onrender.com";
  })(),
  MAX_FILE_SIZE_MB: 10,
};
