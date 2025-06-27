document.getElementById('newsForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const newsContent = document.getElementById('newsInput').value;

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `news=${encodeURIComponent(newsContent)}`
    });

    const data = await response.json();
    document.getElementById('result').innerText = `This news is: ${data.result}`;
});
