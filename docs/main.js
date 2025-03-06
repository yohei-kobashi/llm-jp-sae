let main = document.querySelector('.main');

// Checkpoint picker (10, 100, 1000, ..., 988240)
var numPicker = new hx.Picker('#feature-picker',{
    items: Array.from({length: 100}, (_, i) => i),
    startValue: 0
}).value(0);

var ckptPicker = new hx.Picker('#ckpt-picker', {
    items: ["10", "100", "1000", "10000", "100000", "988240"],
    startValue: '988240'
}).value('988240');

loadData(ckptPicker.value(), numPicker.value());

// チェックポイント選択時のイベント
ckptPicker.on('change', () => {
    loadData(ckptPicker.value(), numPicker.value());
});

// データインデックス選択時のイベント
numPicker.on('change', () => {
    loadData(ckptPicker.value(), numPicker.value());
});

numPicker.on('input-change', () => {
    loadData(ckptPicker.value(), numPicker.value());
});

function loadData(ckpt, index) {
    let filePath = `./data/${ckpt}/${index}.json`;

    fetch(filePath)
        .then(response => response.json())
        .then(data => {
            visualizeData(data);
        })
        .catch(error => console.error('Error loading data:', error));
}

function visualizeData(data) {
    main.innerHTML = "";
    let allLists = "";

    let en_num = data.language.en;
    let ja_num = data.language.ja;
    let total = en_num + ja_num;
    let language = "Mix";
    if (total > 0) {
        if (en_num / total > 0.9) {
            language = "English";
        } else if (ja_num / total > 0.9) {
            language = "Japanese";
        }
    }

    let gran_num = data.granularity;
    let gran_str = "Not Defined (only defined for 0-99 features)";
    if (gran_num != null) {
        if (gran_num == 1){
            gran_str = "Uninterpretable";
        } else if (gran_num == 2) {
            gran_str = "Concept-level (Semantic Sim.)";
        } else if (gran_num == 3) {
            gran_str = "Concept-level (Synonymy)";
        } else if (gran_num == 4) {
            gran_str = "Token-level";
        }
    }

    allLists += `<p>Language Trend: ${language} (en:ja = ${en_num}:${ja_num})<br>Granularity: ${gran_str}</p>`;

    for (const sentence of data.token_act) {
        allLists += "<p>";
        for (const [token, actValue] of sentence) {
            let listItem = `<span class="q${actValue}_orange">${token}</span>`;
            allLists += listItem;
        }
        allLists += "</p>";
    }
    
    main.innerHTML = allLists;
}
