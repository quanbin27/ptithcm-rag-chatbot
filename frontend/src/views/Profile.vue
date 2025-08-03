<template>
  <div class="profile-layout">
    <!-- Header -->
    <header class="header">
      <div class="header-content">
        <h1>PTITHCM RAG System</h1>
        <nav class="nav">
          <router-link to="/chat" class="nav-link">Chat</router-link>
          <router-link v-if="userRole !== 'student'" to="/documents" class="nav-link">Tài liệu</router-link>
          <router-link to="/profile" class="nav-link active">Hồ sơ</router-link>
          <button @click="logout" class="btn-logout">Đăng xuất</button>
        </nav>
      </div>
    </header>

    <div class="main-content">
      <div class="container">
        <div class="profile-card">
          <h2>Thông tin cá nhân</h2>
          
          <div v-if="user" class="profile-info">
            <div class="info-item">
              <label>Họ và tên:</label>
              <span>{{ user.full_name }}</span>
            </div>
            
            <div class="info-item">
              <label>Email:</label>
              <span>{{ user.email }}</span>
            </div>
            
            <div class="info-item">
              <label>Vai trò:</label>
              <span>{{ getRoleName(user.role) }}</span>
            </div>
            
            <div class="info-item">
              <label>Ngày tham gia:</label>
              <span>{{ formatDate(user.created_at) }}</span>
            </div>
          </div>
          
          <div v-else class="loading">
            <span class="spinner"></span>
            Đang tải thông tin...
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

export default {
  name: 'Profile',
  setup() {
    const router = useRouter()
    const authStore = useAuthStore()
    
    const user = computed(() => authStore.user)
    const userRole = computed(() => user.value?.role)
    
    const getRoleName = (role) => {
      const roleNames = {
        student: 'Sinh viên',
        teacher: 'Giảng viên',
        admin: 'Quản trị viên'
      }
      return roleNames[role] || role
    }
    
    const formatDate = (dateString) => {
      return new Date(dateString).toLocaleDateString('vi-VN', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'Asia/Ho_Chi_Minh'
      })
    }
    
    const logout = () => {
      authStore.logout()
      router.push('/login')
    }
    
    onMounted(async () => {
      // Đảm bảo user được load từ localStorage
      if (authStore.token && !authStore.user) {
        await authStore.initializeAuth()
      } else if (!user.value) {
        await authStore.fetchUser()
      }
    })
    
    return {
      user,
      userRole,
      getRoleName,
      formatDate,
      logout
    }
  }
}
</script>

<style scoped>
.profile-layout {
  min-height: 100vh;
  background: #f5f7fa;
}

.header {
  background: white;
  border-bottom: 1px solid #e1e5e9;
  padding: 16px 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header h1 {
  color: #2c3e50;
  font-size: 24px;
  font-weight: 600;
}

.nav {
  display: flex;
  gap: 20px;
  align-items: center;
}

.nav-link {
  text-decoration: none;
  color: #6c757d;
  font-weight: 500;
  padding: 8px 16px;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.nav-link:hover,
.nav-link.active {
  color: #667eea;
  background: #f8f9fa;
}

.btn-logout {
  background: #e74c3c;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: background-color 0.3s ease;
}

.btn-logout:hover {
  background: #c0392b;
}

.main-content {
  padding: 40px 0;
}

.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 0 20px;
}

.profile-card {
  background: white;
  border-radius: 12px;
  padding: 32px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.profile-card h2 {
  color: #2c3e50;
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 24px;
  text-align: center;
}

.profile-info {
  max-width: 400px;
  margin: 0 auto;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 0;
  border-bottom: 1px solid #e1e5e9;
}

.info-item:last-child {
  border-bottom: none;
}

.info-item label {
  font-weight: 600;
  color: #2c3e50;
  min-width: 120px;
}

.info-item span {
  color: #6c757d;
  text-align: right;
}

.loading {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
  color: #6c757d;
}
</style> 