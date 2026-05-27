<template>
  <section id="scenarios" class="section section-light">
    <div class="section-title">
      <p class="en">SCENARIOS</p>
      <h2>面向大型工程结构应用场景</h2>
      <p class="desc">
        聚焦复杂工程结构在地震、振动及动力荷载作用下的安全性、舒适性与使用性能问题。
      </p>
    </div>

    <div class="scenario-grid">
      <div v-for="item in scenarios" :key="item.title" class="scenario-card">
        <span>{{ item.no }}</span>
        <h3>{{ item.title }}</h3>
        <p>{{ item.desc }}</p>
      </div>
    </div>
  </section>
</template>

<script setup>
const scenarios = [
  { no: '01', title: '机场与航站楼', desc: '面向大跨空间结构与复杂公共建筑，提供隔震减震与动力响应分析支持。' },
  { no: '02', title: '高层建筑', desc: '关注高层结构动力响应、振动控制、支座布置及结构安全性能提升。' },
  { no: '03', title: '桥梁工程', desc: '服务桥梁结构在地震、车辆及环境荷载作用下的隔震与动力控制需求。' },
  { no: '04', title: '轨道交通', desc: '开展轨道交通引起的结构振动响应、振动传播与噪声影响分析。' },
  { no: '05', title: '复杂工业结构', desc: '面向设备振动、厂房结构、平台结构等场景开展振动控制与评估。' },
  { no: '06', title: '特殊结构', desc: '针对复杂边界、复杂荷载和复杂功能要求的结构提供定制化技术支持。' },
]
</script>

<style scoped>
.scenario-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 26px;
}

.scenario-card {
  padding: 32px;
  border-radius: 24px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  transition: 0.25s;
}

.scenario-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 20px 45px rgba(15, 23, 42, 0.1);
}

.scenario-card span {
  color: #0284c7;
  font-weight: 800;
  letter-spacing: 2px;
}

.scenario-card h3 {
  font-size: 24px;
  margin: 18px 0 12px;
}

.scenario-card p {
  color: #475569;
  line-height: 1.8;
}

@media (max-width: 980px) {
  .scenario-grid {
    grid-template-columns: 1fr;
  }
}
</style>