<template>
  <header class="nav">
    <div class="brand">
      <span class="brand-mark"></span>
      <div>
        <strong>震安振动控制事业部</strong>
        <p>Zhenan Vibration Control</p>
      </div>
    </div>

    <nav>
      <a href="#scenarios">应用场景</a>
      <a href="#capability">核心能力</a>
      <a href="#tools">数字化工具</a>
      <a href="#team">团队支撑</a>
      <a href="#contact">联系我们</a>
    </nav>
  </header>
</template>

<style scoped>
.nav {
  height: 76px;
  padding: 0 7%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(7, 17, 31, 0.84);
  backdrop-filter: blur(18px);
  border-bottom: 1px solid rgba(148, 163, 184, 0.18);
  position: sticky;
  top: 0;
  z-index: 50;
}

.brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.brand-mark {
  width: 38px;
  height: 38px;
  border-radius: 12px;
  background: linear-gradient(135deg, #38bdf8, #2563eb);
  box-shadow: 0 0 30px rgba(56, 189, 248, 0.65);
}

.brand strong {
  font-size: 18px;
}

.brand p {
  margin: 3px 0 0;
  font-size: 12px;
  color: #93c5fd;
}

.nav a {
  margin-left: 26px;
  color: #cbd5e1;
  text-decoration: none;
  font-size: 15px;
}

.nav a:hover {
  color: #38bdf8;
}

@media (max-width: 980px) {
  .nav nav {
    display: none;
  }
}
</style>