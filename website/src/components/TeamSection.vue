<template>
  <section id="team" class="section section-dark team">
    <div class="section-title">
      <p class="en">TEAM</p>
      <h2>产学研结合的工程技术团队</h2>
      <p class="desc">
        团队由教授、高级工程师、博士、硕士研究人员与产品设计人员组成，
        形成工程经验、理论分析、数字化开发和产品化设计相结合的综合能力。
      </p>
    </div>

    <div class="team-stats">
      <div>
        <strong>教授</strong>
        <p>理论研究与技术指导</p>
      </div>
      <div>
        <strong>高级工程师</strong>
        <p>多年大型项目经验</p>
      </div>
      <div>
        <strong>博士 / 硕士团队</strong>
        <p>分析计算与工程研发</p>
      </div>
      <div>
        <strong>产品设计</strong>
        <p>工具平台与产品化支持</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.team {
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 35%),
    #07111f;
}

.team-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 22px;
}

.team-stats div {
  padding: 30px;
  border-radius: 24px;
  background: rgba(15, 23, 42, 0.76);
  border: 1px solid rgba(125, 211, 252, 0.18);
}

.team-stats strong {
  display: block;
  color: #38bdf8;
  font-size: 24px;
  margin-bottom: 12px;
}

.team-stats p {
  margin: 0;
  color: #cbd5e1;
  line-height: 1.7;
}

@media (max-width: 980px) {
  .team-stats {
    grid-template-columns: 1fr;
  }
}
</style>