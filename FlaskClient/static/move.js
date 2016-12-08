function move() {
    var elem = document.getElementById("bar");
    var scripts = document.getElementsByTagName('script');
    var lastScript = scripts[scripts.length-1];
    var x = parseInt(lastScript.getAttribute('data-clickbait'));
    var width = 0;
    var increment = x/24;
    var id = setInterval(frame, 24);
    function frame() {
        if (width >= x) {
            clearInterval(id);
        } else {
            width = width + increment;
            elem.style.width = width + '%';
        }
    }
    elem.style.width = x;
}
window.onload=move();