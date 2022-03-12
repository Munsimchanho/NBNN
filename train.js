let data = [];

let layerCount = 4;
let neuronCount = [28*28, 16, 16, 10];
let iter = 50;

let w = [];
let a = [];
let z = [];
let b = [];
let da = [];
let dw = [];
let db = [];

$.ajax({
    url: './mnist_train/mnist_train_to50000.csv',
    dataType:'text',
    timeout: 2*60*60*1000,
    success: function(res) {
        let allRow = res.split("\n");
        for(let singleRow = 1; singleRow < allRow.length; singleRow++) {
            data[singleRow-1] = allRow[singleRow].split(",");
            //console.log(data[singleRow-1]);
        }
                
        for (let L = 0; L < layerCount; L++){
            // a, z, da 설정
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
                
                // b 설정
                // Math.random() : 0 ~ 1                
                b[L].push(Math.random() * 2 - 1);                
                
                w[L].push([]);
                dw[L].push([]);
                db[L].push([]);
                for (let i = 0; i < neuronCount[L - 1]; i++){
                    w[L][j].push(Math.random() * 2 - 1);
                }
            }
        }
        
        for (let i = 0; i < data.length - 1; i++){
            ff(data[i]);
            bp();
            
            //console.log(w[layerCount - 1][0]);
            //console.log(a[layerCount - 1]);
            //console.log(Cost());
            //console.log("");
        }  

        console.log(JSON.stringify(w));
        console.log(JSON.stringify(b));
    }
});

// 출력용
function Cost(){
    let sum = 0;
    let costStr = '';
    for (let i = 0; i < 10; i++){
        sum += (y[i] - a[layerCount - 1][i]) * (y[i] - a[layerCount - 1][i]);
        costStr += (y[i] - a[layerCount - 1][i]) * (y[i] - a[layerCount - 1][i]) + ' ';
    }
    //console.log(costStr);
    return sum / 10;
}  

// 출력용
function printArr(arr){
    let arrText = '';
    for (var i = 2; i < 3; i++) {
        for (var j = 0; j < arr[i].length; j++) {
            arrText+=arr[i][j]+'     ';
        }
        console.log(arrText);
        arrText='';
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

let y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const learningRate = 1;

// 개수를 변수로 저장하고 w, b를 랜덤하게 배치

// L번째 layer의 모든 neuron에 대해 값 변경 && 편미분 값 초기화
function updateLayer(L){
    for (let i = 0; i < neuronCount[L]; i++){
        // z 계산
        
        //console.log(w[L][i]);
        //console.log(a[L - 1]);
        z[L][i] = dot(w[L][i], a[L - 1]) + b[L][i];
        //console.log(z[L][i]);
        a[L][i] = sigmoid(z[L][i]);
        //console.log(z[L][i]);
        da[L][i] = null;
        db[L][i] = null;
        for (let j = 0; j < neuronCount[L - 1]; j++){
            dw[L][i][j] = null;
        }
    }
    
    //console.log(a[L]);
}

// 순전파 (Feed Forward)
function ff(input){
    // 첫 번째 layer에 대해
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    y[parseInt(input[0])] = 1;
    //console.log(a);
    
    for (let i = 0; i < neuronCount[0]; i++){
        // i + 1인 이유는 input[0]이 label이라서
        a[0][i] = parseInt(input[i + 1]) / 255;
    }
    
    // 모든 layer에 대해
    for (let L = 1; L < layerCount; L++){
        updateLayer(L);
    }
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
            b[L][j] -= learningRate * db[L][j];
            
            for (let i = 0; i < neuronCount[L - 1]; i++){
                w[L][j][i] -= learningRate * dw[L][j][i];
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

