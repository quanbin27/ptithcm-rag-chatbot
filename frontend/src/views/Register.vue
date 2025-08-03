<template>
  <div class="auth-container">
    <div class="auth-card">
      <h1 class="auth-title">Đăng ký</h1>
      
      <form @submit.prevent="handleRegister">
        <div class="form-group">
          <label for="fullName">Họ và tên</label>
          <input
            id="fullName"
            v-model="form.full_name"
            type="text"
            required
            placeholder="Nhập họ và tên"
          />
        </div>
        
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
            minlength="6"
          />
        </div>
        
        <div class="form-group">
          <label for="role">Vai trò</label>
          <select
            id="role"
            v-model="form.role"
            required
            class="form-select"
          >
            <option value="student">Sinh viên</option>
            <option value="teacher">Giảng viên</option>
            <option value="admin">Quản trị viên</option>
          </select>
        </div>
        
        <button 
          type="submit" 
          class="btn-submit"
          :disabled="loading"
        >
          <span v-if="loading" class="spinner"></span>
          {{ loading ? 'Đang đăng ký...' : 'Đăng ký' }}
        </button>
      </form>
      
      <div v-if="error" class="error-message">
        {{ error }}
      </div>
      
      <div v-if="success" class="success-message">
        {{ success }}
      </div>
      
      <div class="auth-footer">
        Đã có tài khoản? 
        <router-link to="/login">Đăng nhập</router-link>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

export default {
  name: 'Register',
  setup() {
    const router = useRouter()
    const authStore = useAuthStore()
    
    const form = reactive({
      full_name: '',
      email: '',
      password: '',
      role: 'student'
    })
    
    const loading = ref(false)
    const error = ref('')
    const success = ref('')
    
    const handleRegister = async () => {
      loading.value = true
      error.value = ''
      success.value = ''
      
      const result = await authStore.register(form)
      
      if (result.success) {
        success.value = 'Đăng ký thành công! Vui lòng đăng nhập.'
        setTimeout(() => {
          router.push('/login')
        }, 2000)
      } else {
        error.value = result.error
      }
      
      loading.value = false
    }
    
    return {
      form,
      loading,
      error,
      success,
      handleRegister
    }
  }
}
</script>

<style scoped>
.form-select {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  font-size: 16px;
  transition: border-color 0.3s ease;
  background: white;
}

.form-select:focus {
  outline: none;
  border-color: #667eea;
}

.error-message {
  color: #e74c3c;
  text-align: center;
  margin-top: 16px;
  padding: 12px;
  background: #fdf2f2;
  border-radius: 8px;
  border: 1px solid #fecaca;
}

.success-message {
  color: #27ae60;
  text-align: center;
  margin-top: 16px;
  padding: 12px;
  background: #f0f9ff;
  border-radius: 8px;
  border: 1px solid #b3e5fc;
}
</style> 