<template>
  <div class="auth-container">
    <div class="auth-card">
      <h1 class="auth-title">Đăng nhập</h1>
      
      <form @submit.prevent="handleLogin">
        <div class="form-group">
          <label for="email">Email</label>
          <input
            id="email"
            v-model="form.email"
            type="email"
            required
            placeholder="Nhập email của bạn"
          />
        </div>
        
        <div class="form-group">
          <label for="password">Mật khẩu</label>
          <input
            id="password"
            v-model="form.password"
            type="password"
            required
            placeholder="Nhập mật khẩu"
          />
        </div>
        
        <button 
          type="submit" 
          class="btn-submit"
          :disabled="loading"
        >
          <span v-if="loading" class="spinner"></span>
          {{ loading ? 'Đang đăng nhập...' : 'Đăng nhập' }}
        </button>
      </form>
      
      <div v-if="error" class="error-message">
        {{ error }}
      </div>
      
      <div class="auth-footer">
        Chưa có tài khoản? 
        <router-link to="/register">Đăng ký ngay</router-link>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

export default {
  name: 'Login',
  setup() {
    const router = useRouter()
    const authStore = useAuthStore()
    
    const form = reactive({
      email: '',
      password: ''
    })
    
    const loading = ref(false)
    const error = ref('')
    
    const handleLogin = async () => {
      loading.value = true
      error.value = ''
      
      const result = await authStore.login(form)
      
      if (result.success) {
        router.push('/chat')
      } else {
        error.value = result.error
      }
      
      loading.value = false
    }
    
    return {
      form,
      loading,
      error,
      handleLogin
    }
  }
}
</script>

<style scoped>
.error-message {
  color: #e74c3c;
  text-align: center;
  margin-top: 16px;
  padding: 12px;
  background: #fdf2f2;
  border-radius: 8px;
  border: 1px solid #fecaca;
}
</style> 