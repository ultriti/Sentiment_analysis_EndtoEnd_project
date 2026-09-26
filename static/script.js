const textInput = document.getElementById("textInput");
const analyzeBtn = document.getElementById("analyzeBtn");

const charCount = document.getElementById("charCount");

const resultSection = document.getElementById("resultSection");
const emotionLabel = document.getElementById("emotionLabel");
const emotionEmoji = document.getElementById("emotionEmoji");
const confidenceValue = document.getElementById("confidenceValue");

const probabilityList = document.getElementById("probabilityList");

const errorBox = document.getElementById("errorBox");
const errorMessage = document.getElementById("errorMessage");


// ==========================
// Character counter
// ==========================

textInput.addEventListener("input", () => {

    const length = textInput.value.length;

    charCount.textContent = `${length} / 2000`;

});


// ==========================
// Emotion emojis
// ==========================

const emotionEmojis = {

    sadness: "😢",
    joy: "😄",
    love: "❤️",
    anger: "😠",
    fear: "😨",
    surprise: "😯"

};


// ==========================
// Show error
// ==========================

function showError(message) {

    errorMessage.textContent = message;

    errorBox.classList.remove("hidden");

}


// ==========================
// Hide error
// ==========================

function hideError() {

    errorBox.classList.add("hidden");

}


// ==========================
// Display probabilities
// ==========================

function displayProbabilities(probabilities) {

    probabilityList.innerHTML = "";

    Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1])
        .forEach(([emotion, probability]) => {

            const percentage = probability * 100;

            const row = document.createElement("div");

            row.className = "probability-row";

            row.innerHTML = `
                <div class="probability-label">
                    <span>${emotion}</span>
                    <span>${percentage.toFixed(1)}%</span>
                </div>

                <div class="progress">
                    <div
                        class="progress-bar"
                        style="width: ${percentage}%"
                    ></div>
                </div>
            `;

            probabilityList.appendChild(row);

        });

}


// ==========================
// Analyze text
// ==========================

async function analyzeEmotion() {

    const text = textInput.value.trim();

    hideError();

    if (!text) {

        showError("Please enter some text to analyze.");

        textInput.focus();

        return;
    }


    // Loading state

    analyzeBtn.disabled = true;

    analyzeBtn.innerHTML = `
        Analyzing...
    `;


    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text
            })

        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail || "Unable to analyze the text."
            );

        }


        // ==========================
        // Update result
        // ==========================

        const emotion = data.prediction_emotion;

        const confidence = data.confidence * 100;


        emotionLabel.textContent = emotion;

        emotionEmoji.textContent =
            emotionEmojis[emotion] || "🧠";

        confidenceValue.textContent =
            `${confidence.toFixed(1)}%`;


        displayProbabilities(
            data.all_probabilities
        );


        resultSection.classList.remove("hidden");


        // Scroll result into view

        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });


    } catch (error) {

        console.error(error);

        showError(
            error.message ||
            "Something went wrong while analyzing the text."
        );

    } finally {

        analyzeBtn.disabled = false;

        analyzeBtn.innerHTML = `
            Analyze Emotion
            <span>→</span>
        `;

    }

}


// ==========================
// Button click
// ==========================

analyzeBtn.addEventListener(
    "click",
    analyzeEmotion
);


// ==========================
// Ctrl + Enter shortcut
// ==========================

textInput.addEventListener("keydown", (event) => {

    if (event.ctrlKey && event.key === "Enter") {

        analyzeEmotion();

    }

});