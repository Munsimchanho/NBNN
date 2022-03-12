let inputPar = document.getElementById("input-parent");

let inputValue = [];
let isDrawing = false;

for (let i = 0; i < 28; i++){
    inputValue.push([]);
    let inputLine = document.createElement("div");
    inputLine.className = "input-line";
    inputLine.draggable = false;
    for (let j = 0; j < 28; j++){
        inputValue[28 * i + j] = 0;
        let input = document.createElement("div");
        input.className = `input`;
        input.id = `input-${i}-${j}`;
        input.style.display = "inline-block";
        input.style.background = "#555";
        input.style.opacity = 0;
        input.draggable = false;
        input.style.userSelect = "none";
        
        input.addEventListener("mouseover", () => {
            if (!isDrawing) return;
            
            inputValue[28 * i + j] = 1;
            
            changeOpacity(i, j, 1);
            
            changeOpacity(i - 1, j, .4);
            changeOpacity(i + 1, j, .4);
            changeOpacity(i, j - 1, .4);
            changeOpacity(i, j + 1, .4);
        });
        
        inputLine.appendChild(input);
    }
    
    inputPar.appendChild(inputLine);
}

function changeOpacity(i, j, dv){
    if (i <= 0 || i >= 27 || j <= 0 || j >= 27) return;
    
    inputValue[28 * i + j] = Math.max(Math.min(inputValue[28 * i + j] + dv, 1), 0);
    document.getElementById(`input-${i}-${j}`).style.opacity = inputValue[28 * i + j];
}


document.addEventListener("mousedown", () => {
    isDrawing = true;
});

document.addEventListener("mouseup", () => {
    isDrawing = false;
});

let w = [];
let a = [];
let z = [];
let b = [];
let da = [];
let dw = [];
let db = [];

let layerCount = 4;
let neuronCount = [28*28, 16, 16, 10];

$.ajax({
    url: './trained_data/w.json',
    success: function(_w) {
        $.ajax({
            url: './trained_data/b.json',
            success: function(_b){
                w = _w;
                b = _b;
                                
                for (let L = 0; L < layerCount; L++){
                    a.push([]);
                    z.push([]);
                    da.push([]);
                    dw.push([]);
                    db.push([]);
                    b.push([]);
                    w.push([]);

                    if (L == 0) continue;
                    // w, b 설정
                    for (let j = 0; j < neuronCount[L]; j++){
                        dw[L].push([]);
                        db[L].push([]);
                    }
                }
            }
        })
    }
});

let y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

let guess;
function submit(){
    guess = ff(inputValue);
    console.log(guess);
}

function update(corr, val){
    if (corr){
        val = guess;
    }
    
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    y[val] = 1;
    
    bp();
    
    let formData = new FormData();
    formData.append("code", "update_data");
    formData.append("wdata", JSON.stringify(w));
    formData.append("bdata", JSON.stringify(b));
    $.ajax({
        url         : "./server.php",
        type        : "POST",
        dataType    : 'html',
        enctype     : "multipart/form-data",
        processData : false,
        contentType : false,
        data        : formData,
        async       : false,
        success     : function(res){ 
            console.log(res);
        }
    });
    
    //document.getElementById("number-input").value = "";
    reset();
}

function reset(){
    for (let i = 0; i < 28; i++){
        for (let j = 0; j < 28; j++){
            changeOpacity(i, j, -999);
        }
    }
}

function ff(val){
    for (let i = 0; i < neuronCount[0]; i++){
        a[0][i] = val[i];
    }
    
    // 모든 layer에 대해
    for (let L = 1; L < layerCount; L++){
        updateLayer(L);
    }
    
    let max = -987654321;
    let index = 0;
    
    console.log(a[layerCount - 1]);
    
    // 결과 반환
    for (let i = 0; i < neuronCount[layerCount - 1]; i++){
        if (a[layerCount - 1][i] > max){
            index = i;
            max = a[layerCount - 1][i];
        }
    }
    
    return index;
}

// L번째 layer의 모든 neuron에 대해 값 변경 && 편미분 값 초기화
function updateLayer(L){
    for (let i = 0; i < neuronCount[L]; i++){
        // z, a 계산
        z[L][i] = dot(w[L][i], a[L - 1]) + b[L][i];
        a[L][i] = sigmoid(z[L][i]);
        
        da[L][i] = null;
        db[L][i] = null;
        for (let j = 0; j < neuronCount[L - 1]; j++){
            dw[L][i][j] = null;
        }
    }
}

// 벡터 내적 계산
function dot(m1, m2){
    let sum = 0;
    for (let i = 0; i < m1.length; i++){
        sum += m1[i] * m2[i];
    }
    return sum;
}

// 시그모이드 함수
function sigmoid(x){
    return 1 / (1 + Math.exp(-x));
}

// 역전파 (Back Propagation)
function bp(){
    // 비용함수의 w에 대한 편미분 값(dC/dw) 계산
    for (let L = 1; L < layerCount; L++){
        for (let j = 0; j < neuronCount[L]; j++){
            for (let i = 0; i < neuronCount[L - 1]; i++){
                dw[L][j][i] = bp_w(L, j, i);
            }
        }
    }
    
    // 비용함수의 b에 대한 편미분 값(dC/db) 계산
    for (let L = 1; L < layerCount; L++){
        for (let i = 0; i < neuronCount[L]; i++){
            db[L][i] = bp_b(L, i);
        }
    }
    
    // 편미분 값 적용
    for (let L = 1; L < layerCount; L++){
        for (let j = 0; j < neuronCount[L]; j++){
            b[L][j] -= db[L][j];
            
            for (let i = 0; i < neuronCount[L - 1]; i++){
                w[L][j][i] -= dw[L][j][i];
            }
        }
    }
}

// L번째 layer의 j번째 neuron과 L-1번째 layer의 i번째 neuron을 잇는 간선의 가중치 (편미분 값) 계산
function bp_w(L, j, i){
    if (da[L][j] == null) da[L][j] = bp_a(L, j);
    
    //console.log(da[L][j] * a[L][j] * (1 - a[L][j]) * a[L - 1][i]);
    
    return da[L][j] * a[L][j] * (1 - a[L][j]) * a[L - 1][i];
}

// L번째 layer의 i번째 neuron에 대한 편미분 값 계산
function bp_a(L, i){
    // base case
    if (L == layerCount - 1) return .2 * (a[L][i] - y[i]);

    let sum = 0;
    for (let j = 0; j < neuronCount[L + 1]; j++){
        if (da[L + 1][j] == null) da[L + 1][j] = bp_a(L + 1, j);
        sum += da[L + 1][j] * a[L + 1][j] * (1 - a[L + 1][j]) * w[L + 1][j][i];
    }
    return sum;
}

// L번째 layer의 i번째 neuron의 bias에 대한 편미분 값 계산
function bp_b(L, i){
    if (da[L][i] == null) da[L][i] = bp_a(L, i);
    return da[L][i] * a[L][i] * (1 - a[L][i]);
}