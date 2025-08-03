<template>
  <div class="chat-layout">
    <!-- Header -->
    <header class="header">
      <div class="header-content">
        <h1>PTITHCM RAG System</h1>
        <nav class="nav">
          <router-link to="/chat" class="nav-link active">Chat</router-link>
          <router-link v-if="userRole !== 'student'" to="/documents" class="nav-link">Tài liệu</router-link>
          <router-link to="/profile" class="nav-link">Hồ sơ</router-link>
          <button @click="logout" class="btn-logout">Đăng xuất</button>
        </nav>
      </div>
    </header>

    <div class="main-content">
      <!-- Sidebar -->
      <aside class="sidebar">
        <div class="sidebar-header">
          <h3>Lịch sử chat</h3>
          <button @click="createNewChat" class="btn-new-chat">
            <el-icon><Plus /></el-icon>
            Chat mới
          </button>
        </div>
        
        <div class="sessions-list">
          <div
            v-for="session in sessions"
            :key="session.id"
            :class="['session-item', { active: currentSession?.id === session.id }]"
            @click="selectSession(session)"
          >
            <div class="session-title">{{ session.title }}</div>
            <div class="session-date">{{ formatDate(session.updated_at) }}</div>
          </div>
        </div>
      </aside>

      <!-- Chat Area -->
      <main class="chat-area">
        <div class="chat-container">
          <div class="chat-messages" ref="messagesContainer">
            <div v-if="!currentSession" class="welcome-message">
              <h2>Chào mừng đến với PTITHCM RAG System!</h2>
              <p>Hãy bắt đầu cuộc trò chuyện mới hoặc chọn một cuộc trò chuyện từ lịch sử.</p>
            </div>
            
            <div
              v-for="message in messages"
              :key="message.message_id"
              :class="['message', message.role]"
            >
              <div class="message-content">
                <div v-if="message.role === 'assistant'" class="message-header">
                  <el-icon><ChatDotRound /></el-icon>
                  PTITHCM Assistant
                </div>
                <div class="message-text" v-html="formatMessage(message.content)"></div>
                <div v-if="message.sources && message.sources.length > 0" class="message-sources">
                  <div class="sources-title">Nguồn tham khảo:</div>
                  <div v-for="source in getUniqueSources(message.sources)" :key="source.metadata?.source" class="source-item">
                    {{ source.metadata?.source || 'Tài liệu' }}
                  </div>
                </div>
              </div>
            </div>
            
            <div v-if="loading" class="message assistant">
              <div class="message-content">
                <div class="message-header">
                  <el-icon><ChatDotRound /></el-icon>
                  PTITHCM Assistant
                </div>
                <div class="loading-message">
                  <span class="spinner"></span>
                  Đang xử lý...
                </div>
              </div>
            </div>
          </div>

          <div class="chat-input">
            <input
              v-model="newMessage"
              @keyup.enter="sendMessage"
              placeholder="Nhập câu hỏi của bạn..."
              :disabled="loading"
            />
            <button @click="sendMessage" :disabled="loading || !newMessage.trim()" class="btn-send">
              <el-icon><Promotion /></el-icon>
            </button>
          </div>
        </div>
      </main>
    </div>
  </div>
</template>

<script>
import { ref, reactive, onMounted, nextTick, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { ElMessage } from 'element-plus'
import api from '@/services/api'
import { marked } from 'marked'

export default {
  name: 'Chat',
  setup() {
    const router = useRouter()
    const authStore = useAuthStore()
    
    const sessions = ref([])
    const currentSession = ref(null)
    const messages = ref([])
    const newMessage = ref('')
    const loading = ref(false)
    const messagesContainer = ref(null)
    
    const fetchSessions = async () => {
      try {
        const response = await api.get('/chat/sessions')
        sessions.value = response.data
      } catch (error) {
        ElMessage.error('Không thể tải lịch sử chat')
      }
    }
    
    const fetchMessages = async (sessionId) => {
      try {
        const response = await api.get(`/chat/sessions/${sessionId}/messages`)
        messages.value = response.data
        await nextTick()
        scrollToBottom()
      } catch (error) {
        ElMessage.error('Không thể tải tin nhắn')
      }
    }
    
    const sendMessage = async () => {
      if (!newMessage.value.trim() || loading.value) return
      
      const message = newMessage.value
      newMessage.value = ''
      loading.value = true
      
      try {
        const response = await api.post('/chat/send', {
          message,
          session_id: currentSession.value?.id
        })
        
        // Update current session
        currentSession.value = { id: response.data.session_id }
        
        // Add user message
        messages.value.push({
          message_id: Date.now().toString(),
          content: message,
          role: 'user',
          timestamp: new Date().toISOString()
        })
        
        // Add assistant message
        messages.value.push({
          message_id: response.data.message_id,
          content: response.data.response,
          role: 'assistant',
          timestamp: new Date().toISOString(),
          sources: response.data.sources
        })
        
        await nextTick()
        scrollToBottom()
        
        // Refresh sessions
        await fetchSessions()
        
      } catch (error) {
        ElMessage.error('Không thể gửi tin nhắn')
        newMessage.value = message // Restore message
      } finally {
        loading.value = false
      }
    }
    
    const createNewChat = () => {
      currentSession.value = null
      messages.value = []
    }
    
    const selectSession = async (session) => {
      currentSession.value = session
      await fetchMessages(session.id)
    }
    
    const scrollToBottom = () => {
      if (messagesContainer.value) {
        messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
      }
    }
    
    const formatMessage = (content) => {
      return marked(content)
    }
    
    const formatDate = (dateString) => {
      // Chuyển đổi từ UTC sang giờ Việt Nam (+7)
      const utcDate = new Date(dateString);
      const vietnamTime = new Date(utcDate.getTime() + (7 * 60 * 60 * 1000));
      
      return vietnamTime.toLocaleDateString('vi-VN', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    }

    const getUniqueSources = (sources) => {
      const seenSources = new Set();
      const uniqueSources = [];
      
      sources.forEach(source => {
        const sourceName = source.metadata?.source || 'Tài liệu';
        if (!seenSources.has(sourceName)) {
          seenSources.add(sourceName);
          uniqueSources.push(source);
        }
      });
      
      return uniqueSources.slice(0, 3); // Giới hạn 3 nguồn
    };
    
    const userRole = computed(() => {
      const role = authStore.user?.role || null
      console.log('Current user role:', role)
      return role
    })
    
    const logout = () => {
      authStore.logout()
      router.push('/login')
    }
    
    onMounted(async () => {
      // Đảm bảo user được load từ localStorage
      if (authStore.token && !authStore.user) {
        await authStore.initializeAuth()
      }
      
      fetchSessions()
    })
    
    // Watch for user changes
    watch(() => authStore.user, (newUser) => {
      console.log('User changed:', newUser?.role)
    }, { immediate: true })
    
    return {
      sessions,
      currentSession,
      messages,
      newMessage,
      loading,
      messagesContainer,
      sendMessage,
      createNewChat,
      selectSession,
      formatMessage,
      formatDate,
      logout,
      userRole,
      getUniqueSources
    }
  }
}
</script>

<style scoped>
.chat-layout {
  height: 100vh;
  display: flex;
  flex-direction: column;
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
  flex: 1;
  display: flex;
  overflow: hidden;
}

.sidebar {
  width: 300px;
  background: white;
  border-right: 1px solid #e1e5e9;
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 20px;
  border-bottom: 1px solid #e1e5e9;
}

.sidebar-header h3 {
  margin-bottom: 16px;
  color: #2c3e50;
}

.btn-new-chat {
  width: 100%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 12px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  transition: all 0.3s ease;
}

.btn-new-chat:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.sessions-list {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.session-item {
  padding: 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.3s ease;
  margin-bottom: 8px;
  border: 1px solid #e1e5e9;
}

.session-item:hover {
  background-color: #f8f9fa;
}

.session-item.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: #667eea;
}

.session-title {
  font-weight: 500;
  margin-bottom: 4px;
}

.session-date {
  font-size: 12px;
  opacity: 0.8;
}

.chat-area {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 12px;
  margin-bottom: 20px;
}

.welcome-message {
  text-align: center;
  padding: 40px 20px;
  color: #6c757d;
}

.welcome-message h2 {
  color: #2c3e50;
  margin-bottom: 16px;
}

.message {
  margin-bottom: 16px;
  display: flex;
  align-items: flex-start;
}

.message.user {
  justify-content: flex-end;
}

.message-content {
  max-width: 70%;
  padding: 16px;
  border-radius: 18px;
  word-wrap: break-word;
  background: white;
  border: 1px solid #e1e5e9;
  border-bottom-left-radius: 4px;
}

.message.user .message-content {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-bottom-right-radius: 4px;
  border-bottom-left-radius: 18px;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  margin-bottom: 8px;
  font-size: 14px;
}

.message-text {
  line-height: 1.6;
}

.message-text :deep(p) {
  margin-bottom: 8px;
}

.message-text :deep(ul), .message-text :deep(ol) {
  margin-left: 20px;
  margin-bottom: 8px;
}

.message-sources {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

.sources-title {
  font-size: 12px;
  font-weight: 500;
  margin-bottom: 4px;
  opacity: 0.8;
}

.source-item {
  font-size: 12px;
  opacity: 0.7;
  margin-bottom: 2px;
}

.loading-message {
  display: flex;
  align-items: center;
  gap: 8px;
}

.chat-input {
  display: flex;
  gap: 12px;
  align-items: center;
}

.chat-input input {
  flex: 1;
  padding: 16px 20px;
  border: 2px solid #e1e5e9;
  border-radius: 24px;
  font-size: 16px;
  transition: border-color 0.3s ease;
}

.chat-input input:focus {
  outline: none;
  border-color: #667eea;
}

.btn-send {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 16px;
  border-radius: 50%;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
}

.btn-send:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.btn-send:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style> 