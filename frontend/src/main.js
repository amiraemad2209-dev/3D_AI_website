import "./style.css";
import * as THREE from "three";

window.addEventListener("DOMContentLoaded", () => {

  // =====================
  // SCENE
  // =====================

  const scene = new THREE.Scene();
  const clock = new THREE.Clock();

  // =====================
  // CAMERA
  // =====================

  const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );

  camera.position.z = 7;

  // =====================
  // RENDERER
  // =====================

  const renderer = new THREE.WebGLRenderer({
    antialias: true
  });

  renderer.setSize(window.innerWidth, window.innerHeight);

  document
    .getElementById("app")
    .appendChild(renderer.domElement);

  // =====================
  // STARS
  // =====================

  const starsGeometry = new THREE.BufferGeometry();

  const starsCount = 5000;

  const positions = new Float32Array(starsCount * 3);

  for (let i = 0; i < starsCount * 3; i++) {
    positions[i] = (Math.random() - 0.5) * 2000;
  }

  starsGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(positions, 3)
  );

  const stars = new THREE.Points(
    starsGeometry,
    new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.7,
    })
  );

  scene.add(stars);

  // =====================
  // CORE
  // =====================

  const core = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1, 1),
    new THREE.MeshPhongMaterial({
      color: 0xffffff,
      wireframe: true,
      transparent: true,
      opacity: 0.17,
    })
  );

  core.scale.set(5, 5, 5);
  scene.add(core);

  // =====================
  // LIGHTS
  // =====================

  scene.add(new THREE.AmbientLight(0x5F7BBA, 2));

  const sunLight = new THREE.PointLight(0xffffff, 33);
  scene.add(sunLight);

  // =====================
  // ANIMATION
  // =====================

  function animate() {
    requestAnimationFrame(animate);

    const t = clock.getElapsedTime();

    stars.rotation.y += 0.0002;

    core.rotation.x += 0.001;
    core.rotation.y += 0.001;

    sunLight.position.x = Math.sin(t) * 8;
    sunLight.position.y = Math.cos(t) * 8;

    const pulse = 1 + Math.sin(t * 2) * 0.03;

    core.scale.set(
      5 * pulse,
      5 * pulse,
      5 * pulse
    );

    renderer.render(scene, camera);
  }

  animate();

  // =====================
  // RESIZE
  // =====================

  window.addEventListener("resize", () => {
    camera.aspect =
      window.innerWidth / window.innerHeight;

    camera.updateProjectionMatrix();

    renderer.setSize(
      window.innerWidth,
      window.innerHeight
    );
  });

  // =====================
  // SIDEBAR
  // =====================

  const sidebar = document.getElementById("sidebar");
  const menuToggle = document.getElementById("menu-toggle");
  const closeBtn = document.getElementById("close-sidebar");
  const overlay = document.getElementById("overlay");

  function openSidebar() {
    sidebar?.classList.add("active");
    overlay?.classList.add("active");
    menuToggle?.classList.add("hidden");
  }

  function closeSidebar() {
    sidebar?.classList.remove("active");
    overlay?.classList.remove("active");
    menuToggle?.classList.remove("hidden");
  }

  menuToggle?.addEventListener("click", openSidebar);
  closeBtn?.addEventListener("click", closeSidebar);
  overlay?.addEventListener("click", closeSidebar);


const authBtns = document.querySelectorAll(".auth-btn, .auth-btn-logout");

authBtns.forEach((btn) => {

  // عند الضغط
  btn.addEventListener("click", () => {

    // نشيل active من الكل
    authBtns.forEach(b => b.classList.remove("active"));

    // نضيف active للزرار المضغوط
    btn.classList.add("active");

  });

});


// =====================
// HERO SCROLL EFFECT
// =====================

const heroSection = document.querySelector(".page-section");

window.addEventListener("scroll", () => {
  const triggerPoint = window.innerHeight * 0.4;

  if (window.scrollY > triggerPoint) {
    heroSection?.classList.add("scrolled");
  } else {
    heroSection?.classList.remove("scrolled");
  }
});
  
// =====================
// CHAT SYSTEM
// =====================

const chatContainer = document.getElementById("chat-container");
const chatHeader = document.getElementById("chat-header");
const minimizeBtn = document.getElementById("minimize-chat");

const input = document.getElementById("input");
const messages = document.getElementById("messages");
const sendBtn = document.getElementById("send");

// =====================
// AUTO RESIZE INPUT
// =====================

function autoResizeInput() {
  if (!input) return;

  input.style.height = "45px";
  input.style.height = input.scrollHeight + "px";
}

input?.addEventListener("input", autoResizeInput);

function autoResizeChat() {
  if (!chatContainer || !messages) return;

  // ❌ لو الشات متصغر، ما نعملش resize
  if (!chatContainer.classList.contains("active")) return;

  const base = 65;
  const inputHeight = 90;
  const max = 390;
  const min = 210;

  let height = base + inputHeight + messages.scrollHeight;

  height = Math.max(min, Math.min(max, height));

  chatContainer.style.height = height + "px";

  messages.scrollTop = messages.scrollHeight;
}

// =====================
// OBSERVER FOR MESSAGES
// =====================

const observer = new MutationObserver(autoResizeChat);

if (messages) {
  observer.observe(messages, {
    childList: true,
    subtree: true
  });
}

// =====================
// RESET INPUT
// =====================

function resetInput() {
  if (!input) return;

  input.value = "";
  input.style.height = "45px";
}

// =====================
// ADD MESSAGE
// =====================

function addMessage(text, sender) {
  if (!messages) return;

  const msg = document.createElement("div");

  msg.classList.add("msg");

  if (sender === "You") {
    msg.classList.add("user-msg");
  } else {
    msg.classList.add("ai-msg");
  }

  msg.innerHTML = `<strong>${sender}:</strong> ${text}`;

  messages.appendChild(msg);

  messages.scrollTop = messages.scrollHeight;

  autoResizeChat?.();
}

// =====================
// OPEN / CLOSE CHAT
// =====================

chatHeader?.addEventListener("click", () => {
  const isActive = chatContainer.classList.toggle("active");

  if (isActive) {
    minimizeBtn.innerHTML = "—";

    // أول ما يفتح يرجع يتظبط تلقائي
    autoResizeChat();
  } else {
    minimizeBtn.innerHTML = "+";

    // 🔥 يرجع للحجم الأساسي
    chatContainer.style.height = "65px";
  }
});

// =====================
// SEND MESSAGE
// =====================

async function sendMessage() {
  const text = input?.value?.trim();
  if (!text) return;

  addMessage(text, "You");

  try {
    const res = await fetch("http://127.0.0.1:5000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: text
      })
    });

    const data = await res.json();
    addMessage(data.reply, "AI");

  } catch (err) {
    console.error(err);
    addMessage("❌ Backend not reachable", "AI");
  }

  resetInput();
}

// click
sendBtn?.addEventListener("click", sendMessage);

// enter
input?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

});

