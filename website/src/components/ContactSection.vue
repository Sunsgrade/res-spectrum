<template>
  <section id="contact" class="section section-light contact">
    <div class="section-title">
      <p class="en">CONTACT</p>
      <h2>联系我们</h2>
      <p class="desc">
        如需了解大型工程结构动力控制、隔震减震设计、支座优化或数字化分析工具，欢迎联系团队。
      </p>
    </div>

    <div class="contact-card">
      <div>
        <span>电话</span>
        <strong>请填写联系电话</strong>
      </div>
      <div>
        <span>邮箱</span>
        <strong>请填写联系邮箱</strong>
      </div>
      <div>
        <span>微信</span>
        <strong>请填写微信或二维码</strong>
      </div>
      <div>
        <span>地址</span>
        <strong>请填写办公地址</strong>
      </div>
    </div>
  </section>
</template>

<style scoped>
.contact-card {
  max-width: 1000px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 22px;
}

.contact-card div {
  padding: 28px;
  border-radius: 24px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
}

.contact-card span {
  display: block;
  color: #0284c7;
  font-weight: 800;
  margin-bottom: 10px;
}

.contact-card strong {
  color: #0f172a;
  font-size: 20px;
}

@media (max-width: 980px) {
  .contact-card {
    grid-template-columns: 1fr;
  }
}
</style>