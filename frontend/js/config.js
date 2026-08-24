window.SPECTRAPUR_CONFIG = {
  API_BASE_URL: (function () {
    const host = window.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") {
      return "http://localhost:5000";
    }
    // TODO: set this to your deployed Render backend URL.
    return "https://spectrapur-api.onrender.com";
  })(),
  MAX_FILE_SIZE_MB: 10,
};
