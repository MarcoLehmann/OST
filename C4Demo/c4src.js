class C4GameEngine {
    constructor(model, ai) {
        this.model = model;
        this.ai = ai;
        // init with a dummy call:
        this._pick_ai_action();
        this.timer_handle = null;
    }

    play_at(column){
        var added1 = this.model.add(column);
        if (added1) {
            // sleep??
            if (!this.model.terminated) {
                var added2 = false;
                while(!added2) {
                    var a = this._pick_ai_action();
                    added2 = this.model.add(a);
                }
            }
        }
        updateView();
    }

    _pick_ai_action(){
        var b = [this.model.board]; // pack the state into a batch of size 1
        var s = tf.tensor(b, [1, 6, 7]); // then transform the data into a state tensor
        var logits = this.ai.tfmodel.predict(s)
        // console.log("logits=" + logits)
        var a = tf.multinomial(logits, 1).dataSync();
        a = a[0];
        console.log("action=" + a);
        return a;
    }

    self_play(){
        var eng = this;
        function onTimerInterval() {
            if(! eng.model.terminated){
                var a = eng._pick_ai_action(); // same AI plays all the time random actions :-)  ...
                eng.play_at(a);
            }
            else{
                eng.reset();
                eng.timer_handle = window.setInterval(onTimerInterval, 25);
                // window.clearInterval(h);
                // console.log("interval cleared");
            }
        }
        this.timer_handle = window.setInterval(onTimerInterval, 25);
    }

    reset(){
        window.clearInterval(this.timer_handle);
        this.model.reset_model();
        updateView();
    }
}

class C4AI {
    constructor() {
        this.tfmodel = this.build_model();
    }

    build_model(){
        const input = tf.input({shape: [6, 7]});
        const fl = tf.layers.flatten().apply(input);
        const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(fl);
        const dense2 = tf.layers.dense({units: 7}).apply(dense1);
        const model = tf.model({inputs: input, outputs: dense2});
        return model
    }

}

class C4Model {
    constructor() {
        console.log("C4 Constructor")
        this.board = null;
        this.currentPlayer = null;
        this.terminated = null;
        this.winCount1 = 0;
        this.winCount2 = 0;
        this.reset_model();
    }

    reset_model(){
        this.board =
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ];
        this.currentPlayer = 1;
        this.terminated = false;
    }

    _get_row(col) {
        for (var r = 5; r >= 0; r--) {
            if (this.board[r][col] == 0) {
                return r;
            }
        }
        throw "Column is full: c=" + col;
    }

    _board_is_full(){
        var firstRow = this.board[0];
        for (var i = 0; i<7; i++){
            if (firstRow[i] == 0){return false;}
        }
        return true;
    }
    _col_is_full(col) {
        return this.board[0][col] != 0;
    }

    add(col) {
        // returns true if the move was added to the board.
        // check .terminated to see whether the game is done.
        if (this.terminated){
            return false;
        }

        if (this._board_is_full()){
            this.terminated = true;
            return false
        }

        if (this._col_is_full(col)) {
            // console.log("Col is full. c="+col);
            return false;
        }

        var row = this._get_row(col);
        // console.log("add to row:" + row)
        this.board[row][col] = this.currentPlayer;

        var [win, reward, winIdxs] = this.check_win_reward(row, col);
        // console.log("win: " +win + ", R=" + reward);
        // console.log("at location :"+ winIdxs);
        if (win){
            this.terminated = true;
            var winStyle;
            if (this.currentPlayer==1){
                winStyle = "p1_win";
                this.winCount1 += 1;
            }
            else{
                winStyle = "p2_win";
                this.winCount2 += 2
            }
            var c4Board = document.getElementById("c4Board");
            for ( var i = 0; i < 4; i++ ) {
                var cell = c4Board.rows[winIdxs[0][i]].cells[winIdxs[1][i]];
                cell.classList.add(winStyle)
            }
            console.log("game ended. Winner=" + this.currentPlayer)
            }
            else{
                this.currentPlayer = -this.currentPlayer;
            }
        return true;
    }

    check_win_reward(r, c) {
        // returns true if the last move (at location r,c) yields a winning state.
        // console.log("check win");
        // console.log("this.board="+this.board);
        // console.log("this.board[2][5]="+this.board[2][5]);

        var r_min = Math.max(r - 3, 0);
        var r_max = Math.min(r + 3, 5);
        var c_min = Math.max(c - 3, 0);
        var c_max = Math.min(c + 3, 6);

        var n_UL = Math.min(r - r_min, c - c_min);
        var n_UR = Math.min(r - r_min, c_max - c);
        var n_LL = Math.min(r_max - r, c - c_min);
        var n_LR = Math.min(r_max - r, c_max - c);

        var n_diag1 = n_LL + n_UR + 1;
        var n_diag2 = n_UL + n_LR + 1;


        function sliceN(board, startRow, stepRow, startCol, stepCol, nTot){
            var sl = Array(nTot);
            var rowIdxAll = Array(nTot);
            var colIdxAll = Array(nTot);
            var rowIdx;
            var colIdx;
            for (var i=0; i<nTot; i++){
                rowIdx = startRow+ (stepRow*i);
                colIdx = startCol + (stepCol*i)
                rowIdxAll[i] = rowIdx;
                colIdxAll[i] = colIdx;
                sl[i] = board[rowIdx][colIdx];
            }
            var rowColIdx = [rowIdxAll, colIdxAll];
            return [sl, rowColIdx];
        }

        var [state_h, idxAll_h] = sliceN(this.board, r, 0, c_min, 1, c_max-c_min+1);
        var [state_v, idxAll_v] = sliceN(this.board, r_min, 1, c, 0, r_max-r_min+1);
        var [state_diag1, idxAll_d1] = sliceN(this.board,r + n_LL, -1, c - n_LL, 1, n_diag1);
        var [state_diag2, idxAll_d2] = sliceN(this.board,r - n_UL, 1, c - n_UL, 1, n_diag2);

        // console.log("state_h:" + state_h)
        // console.log("idxAll_v:" + idxAll_v)
        // console.log("state_v:" + state_v)
        // console.log("state_diag1:" + state_diag1)
        // console.log("state_diag2:" + state_diag2)

        function _conn4(s, color, rcIdx) {
            if (s.length < 4) {
                return [false, null];
            }
            for (var k = 0; k < s.length - 3; k++) {
                var matchAll = true;
                for (var i = 0; i < 4; i++) {
                    var c = s[k + i];
                    var m = (c == color);
                    matchAll = matchAll && m;
                }
                if (matchAll) {
                    var rowIdx = rcIdx[0].slice(k, k+4);
                    var colIdx = rcIdx[1].slice(k, k+4);
                    var rcWin = [rowIdx, colIdx];
                    return [true, rcWin];
                }
            }
            return [false, null];
        }

        var [win, idxs] = _conn4(state_h,this.currentPlayer, idxAll_h);
        if (win) {
            return [true, 1.0, idxs];
        }
        var [win, idxs] = _conn4(state_v,this.currentPlayer,idxAll_v);
        if (win) {
            return [true, 0.5, idxs];
        }
        var [win, idxs] = _conn4(state_diag1,this.currentPlayer, idxAll_d1);
        if (win) {
            return [true, 2.0, idxs];
        }
        var [win, idxs] = _conn4(state_diag2,this.currentPlayer, idxAll_d2);
        if (win) {
            return [true, 2.0, idxs];
        }
        return [false, 0, null];
    }

}


// view and view-supporting methods:

function generateGrid() {
    console.log("generate Grid")
    var grid = "<table id=\"c4Board\">";
    for ( var row = 1; row <= 6; row++ ) {
        grid += "<tr>";
        for ( var col = 1; col <= 7; col++ ) {
            grid += "<td></td>";
        }
        grid += "</tr>";
    }
    return grid;
}


function initView()
{
    console.log("initView")
    $( "#boardContainer" ).append( this.generateGrid);
    $( "td" ).click(function() {
        var index = $( "td" ).index( this );
        var row = Math.floor( ( index ) / 7) + 1;
        var col = ( index % 7 ) + 1;
        // $( "#msg" ).text( "That was row " + row + " and col " + col );
        c4g.play_at(col-1);
        if (c4g.model.terminated){
            $( "span" ).text( "Game Terminated. Winner:" + c4g.model.currentPlayer );

        }
    });
}

function updateView(){
    var c4Board = document.getElementById("c4Board");
    for ( var r = 0; r < 6; r++ ) {
        for ( var c = 0; c < 7; c++ ) {
            if (c4g.model.board[r][c] == 1){
                var cell = c4Board.rows[r].cells[c];
                cell.classList.add("p1")
            }
            else if (c4g.model.board[r][c] == -1){
                var cell = c4Board.rows[r].cells[c];
                cell.classList.add("p2")
            }
            else {
                var cell = c4Board.rows[r].cells[c];
                cell.className = "";
            }
        }
    }
    $( "#msg" ).text("Wins player 1 / player 2: " + c4m.winCount1 + "/" + c4m.winCount2 );

}

function ai_self_play(){
    // console.log("ai_self_play: c4AI="+c4AI)
    c4g.self_play();
}

function reset_game(){
    console.log("reset game")
    c4g.reset();
}


///////////////////////


const c4m = new C4Model();
const c4AI = new C4AI();
const c4g = new C4GameEngine(c4m, c4AI)
console.log(c4AI.tfmodel);

// var modelSavePath = "Conv1_Feb26_model.tf"
// console.log(`Loading model from ${modelSavePath}...`);
// const m = tf.loadLayersModel(modelSavePath);
// // compileModel(model);
// console.log("model:" + m)
