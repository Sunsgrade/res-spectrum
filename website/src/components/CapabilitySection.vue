<template>
  <section id="capability" class="section section-dark capability">
    <div class="section-title">
      <p class="en">CAPABILITY</p>
      <h2>结构动力控制核心能力</h2>
      <p class="desc">
        围绕大型工程结构的隔震减震设计、支座系统应用、动力响应分析与工程实施，形成完整技术服务能力。
      </p>
    </div>

    <div class="capability-list">
      <div v-for="item in capabilities" :key="item.title" class="capability-card">
        <div class="icon">{{ item.no }}</div>
        <div>
          <h3>{{ item.title }}</h3>
          <p>{{ item.desc }}</p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const capabilities = [
  {
    no: 'A',
    title: '隔震减震设计',
    desc: '针对不同结构体系与工程需求，开展隔震减震方案设计、动力响应评估与技术比选。',
  },
  {
    no: 'B',
    title: '支座选型与布置优化',
    desc: '围绕支座产品性能、结构受力路径与工程布置条件，开展支座选型、数量配置与布置优化。',
  },
  {
    no: 'C',
    title: '结构动力响应分析',
    desc: '开展反应谱分析、时程分析、振动响应分析及相关动力指标计算，为方案设计提供依据。',
  },
  {
    no: 'D',
    title: '工程实施技术支持',
    desc: '结合大型项目实际需求，提供分析计算、方案论证、数据处理、结果解释与工程技术配合。',
  },
]
</script>

<style scoped>
.capability {
  position: relative;
  overflow: hidden;
}

.capability::before {
  content: "";
  position: absolute;
  width: 520px;
  height: 520px;
  right: -180px;
  top: 60px;
  border-radius: 50%;
  background: rgba(56, 189, 248, 0.18);
  filter: blur(80px);
}

.capability-list {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
}

.capability-card {
  display: flex;
  gap: 22px;
  padding: 34px;
  border-radius: 26px;
  background: rgba(15, 23, 42, 0.75);
  border: 1px solid rgba(125, 211, 252, 0.18);
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
}

.icon {
  flex: 0 0 54px;
  height: 54px;
  border-radius: 18px;
  background: linear-gradient(135deg, #38bdf8, #2563eb);
  display: grid;
  place-items: center;
  font-weight: 900;
}

.capability-card h3 {
  margin: 0 0 12px;
  font-size: 24px;
}

.capability-card p {
  margin: 0;
  color: #cbd5e1;
  line-height: 1.8;
}

@media (max-width: 980px) {
  .capability-list {
    grid-template-columns: 1fr;
  }
}
</style>