<template>
  <section class="hero">
    <div class="grid-bg"></div>
    <div class="glow glow-a"></div>
    <div class="glow glow-b"></div>

    <div class="hero-content">
      <p class="eyebrow">STRUCTURAL CONTROL · ISOLATION DESIGN · DIGITAL ENGINEERING</p>

      <h1>
        大型工程结构动力控制<br />
        与隔震解决方案
      </h1>

      <p class="hero-desc">
        面向机场航站楼、高层建筑、桥梁工程、轨道交通及复杂结构，
        聚焦隔震减震设计、支座选型与布置优化、结构动力响应分析，
        提供大型工程结构动力控制整体技术服务。
      </p>

      <div class="hero-actions">
        <a class="btn btn-primary" href="#capability">查看核心能力</a>
        <a class="btn btn-ghost" href="#tools">数字化工具平台</a>
      </div>
    </div>

    <div class="dashboard">
      <div class="panel main-panel">
        <div class="panel-head">
          <span>Isolation & Response Analysis</span>
          <i></i>
        </div>

        <div class="bars">
          <span></span><span></span><span></span><span></span><span></span>
        </div>

        <div class="metrics">
          <div>
            <b>支座</b>
            <p>选型优化</p>
          </div>
          <div>
            <b>结构</b>
            <p>动力分析</p>
          </div>
          <div>
            <b>工程</b>
            <p>数字化支撑</p>
          </div>
        </div>
      </div>

      <div class="panel small-panel">
        <span>Service Focus</span>
        <strong>大型工程</strong>
        <p>机场 · 大楼 · 大桥 · 轨道交通</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.hero {
  min-height: 760px;
  position: relative;
  overflow: hidden;
  padding: 90px 7%;
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  align-items: center;
  gap: 60px;
  background: #07111f;
}

.grid-bg {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(56, 189, 248, 0.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(56, 189, 248, 0.08) 1px, transparent 1px);
  background-size: 54px 54px;
  mask-image: linear-gradient(to bottom, black, transparent 90%);
}

.glow {
  position: absolute;
  width: 420px;
  height: 420px;
  border-radius: 999px;
  filter: blur(90px);
  opacity: 0.42;
}

.glow-a {
  background: #2563eb;
  top: 120px;
  right: 8%;
}

.glow-b {
  background: #06b6d4;
  bottom: 80px;
  left: 5%;
}

.hero-content,
.dashboard {
  position: relative;
  z-index: 2;
}

.eyebrow {
  color: #38bdf8;
  letter-spacing: 3px;
  font-size: 13px;
  font-weight: 800;
}

.hero h1 {
  margin: 22px 0;
  font-size: 62px;
  line-height: 1.15;
  letter-spacing: -2px;
}

.hero-desc {
  max-width: 780px;
  font-size: 20px;
  line-height: 1.9;
  color: #cbd5e1;
}

.hero-actions {
  margin-top: 38px;
}

.hero-actions .btn {
  margin-right: 16px;
}

.dashboard {
  display: grid;
  gap: 20px;
}

.panel {
  border: 1px solid rgba(125, 211, 252, 0.22);
  background: rgba(15, 23, 42, 0.68);
  backdrop-filter: blur(20px);
  box-shadow: 0 25px 80px rgba(0, 0, 0, 0.35);
}

.main-panel {
  height: 370px;
  border-radius: 28px;
  padding: 28px;
}

.panel-head {
  display: flex;
  justify-content: space-between;
  color: #93c5fd;
  font-weight: 700;
}

.panel-head i {
  width: 12px;
  height: 12px;
  background: #22c55e;
  border-radius: 50%;
  box-shadow: 0 0 20px #22c55e;
}

.bars {
  height: 185px;
  margin: 38px 0 28px;
  display: flex;
  align-items: end;
  gap: 18px;
}

.bars span {
  flex: 1;
  border-radius: 999px 999px 0 0;
  background: linear-gradient(to top, #2563eb, #38bdf8);
  box-shadow: 0 0 24px rgba(56, 189, 248, 0.45);
}

.bars span:nth-child(1) { height: 40%; }
.bars span:nth-child(2) { height: 72%; }
.bars span:nth-child(3) { height: 54%; }
.bars span:nth-child(4) { height: 90%; }
.bars span:nth-child(5) { height: 63%; }

.metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 15px;
}

.metrics div {
  padding: 16px;
  border-radius: 18px;
  background: rgba(30, 41, 59, 0.8);
}

.metrics b {
  color: #38bdf8;
  font-size: 22px;
}

.metrics p {
  margin: 6px 0 0;
  color: #94a3b8;
}

.small-panel {
  border-radius: 22px;
  padding: 24px;
}

.small-panel span {
  color: #94a3b8;
}

.small-panel strong {
  display: block;
  margin: 8px 0;
  font-size: 34px;
  color: #38bdf8;
}

.small-panel p {
  margin: 0;
  color: #cbd5e1;
}

@media (max-width: 980px) {
  .hero {
    grid-template-columns: 1fr;
    padding: 76px 5%;
  }

  .hero h1 {
    font-size: 42px;
  }
}
</style>