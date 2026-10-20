
/* HomeMate - Shared JavaScript */

// Run after the page HTML has loaded.
document.addEventListener("DOMContentLoaded", function () {
    console.log("HomeMate dashboard scripts loaded.");

    // Format Indian currency values, for example ₹75,00,000.
    document.querySelectorAll("[data-inr]").forEach(function (element) {
        const amount = Number(element.dataset.inr);

        if (Number.isFinite(amount)) {
            element.textContent = new Intl.NumberFormat("en-IN", {
                style: "currency",
                currency: "INR",
                maximumFractionDigits: 0
            }).format(amount);
        }
    });

    // Prevent accidental multiple submissions on prediction forms.
    document.querySelectorAll("[data-loading-form]").forEach(function (form) {
        form.addEventListener("submit", function () {
            const button = form.querySelector('button[type="submit"]');

            if (button) {
                button.disabled = true;
                button.textContent = "Processing...";
            }
        });
    });
});
