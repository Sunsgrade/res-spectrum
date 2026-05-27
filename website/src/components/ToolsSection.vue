<template>
  <section id="tools" class="section section-light">
    <div class="section-title">
      <p class="en">DIGITAL TOOLS</p>
      <h2>数字化工具与工程插件</h2>
      <p class="desc">
        数字化工具不是主营业务本身，而是团队工程分析、方案优化与项目服务能力的延伸。
      </p>
    </div>

    <div class="tools-grid">
      <div class="tool-panel">
        <p class="label">ONLINE TOOLS</p>
        <h3>在线计算工具</h3>
        <ul>
          <li>反应谱在线计算</li>
          <li>结构动力响应分析</li>
          <li>振动响应计算与结果导出</li>
        </ul>
        <a href="#" class="tool-link">进入在线工具</a>
      </div>

      <div class="tool-panel">
        <p class="label">SAP2000 PLUGINS</p>
        <h3>SAP2000 工程插件</h3>
        <ul>
          <li>轨道交通振动与噪声分析插件</li>
          <li>支座布置优化插件</li>
          <li>结构分析结果自动化处理插件</li>
        </ul>
        <a href="#" class="tool-link">插件下载</a>
      </div>
    </div>
  </section>
</template>

<style scoped>
.tools-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 28px;
}

.tool-panel {
  padding: 40px;
  border-radius: 30px;
  background:
    linear-gradient(135deg, rgba(14, 165, 233, 0.16), rgba(37, 99, 235, 0.08)),
    #f8fafc;
  border: 1px solid #dbeafe;
  box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
}

.label {
  color: #0284c7;
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 2px;
}

.tool-panel h3 {
  font-size: 28px;
  margin: 18px 0;
}

.tool-panel ul {
  padding-left: 20px;
  margin: 0;
  color: #475569;
  line-height: 2;
}

.tool-link {
  display: inline-block;
  margin-top: 28px;
  color: #0369a1;
  font-weight: 800;
  text-decoration: none;
  border-bottom: 2px solid #38bdf8;
  padding-bottom: 5px;
}

@media (max-width: 980px) {
  .tools-grid {
    grid-template-columns: 1fr;
  }
}
</style>