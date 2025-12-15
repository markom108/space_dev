const cursor = document.getElementById("cursor"); // create variable that allows editing object in DOM

const frameWidth = 32;
const frameCount = 1;
let currentFrame = 0;

const animationSpeed = 200; //time [ms] between frames

//animate sprite
// do function, every interval
setInterval(() => {
  currentFrame = (currentFrame + 1) % frameCount;
  cursor.style.backgroundPosition = `-${
    currentFrame * frameWidth
  }px 0px`; /* div-window 32x32 stays in the same place, we are moving only sprint sheet so we want to move it to the righ, thats why "-"*/
}, animationSpeed);

//every time User moves mouse => place out new cursor in place of old one
window.addEventListener('mousemove', (event) => {
  cursor.style.left = event.clientX + 'px'; //edit #cursor{style= } directly in CSS
  cursor.style.top = event.clientY + 'px';
});

//when cursor disappears from our site -> hide custom cursor
document.addEventListener('mouseleave', () => {
  cursor.style.display = 'none';
});

//when cursor reappears on our site -> show custom cursor
document.addEventListener('mouseenter', () => {
  cursor.style.display = 'block';
});


