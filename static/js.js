document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictForm");
    form.addEventListener("submit", async(e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded"
            },
            body: new URLSearchParams(data)
        });

        const html = await response.text();
        document.body.innerHTML = html;
    });
});