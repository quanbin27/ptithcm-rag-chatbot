<template>
  <div class="documents-layout">
    <!-- Header -->
    <header class="header">
      <div class="header-content">
        <h1>PTITHCM RAG System</h1>
        <nav class="nav">
          <router-link to="/chat" class="nav-link">Chat</router-link>
          <router-link to="/documents" class="nav-link active">Tài liệu</router-link>
          <router-link to="/profile" class="nav-link">Hồ sơ</router-link>
          <button @click="logout" class="btn-logout">Đăng xuất</button>
        </nav>
      </div>
    </header>

    <div class="main-content">
      <div class="container">
        <div v-if="!isUserLoaded" class="loading">
          <span class="spinner"></span>
          Đang tải thông tin người dùng...
        </div>
        
        <div v-else>
          <div class="documents-header">
            <h2>Quản lý tài liệu</h2>
            <button 
              v-if="canUpload" 
              @click="showUploadDialog = true" 
              class="btn-upload"
            >
              <el-icon><Upload /></el-icon>
              Tải lên tài liệu
            </button>
            <div v-else class="upload-notice">
              <el-icon><InfoFilled /></el-icon>
              <span>Chỉ giáo viên và quản trị viên mới có thể tải lên tài liệu</span>
            </div>
          </div>

          <!-- Stats -->
          <div class="stats-grid">
            <div class="stat-card">
              <div class="stat-number">{{ stats.total_documents || 0 }}</div>
              <div class="stat-label">Tổng tài liệu</div>
            </div>
            <div class="stat-card">
              <div class="stat-number">{{ stats.total_chunks || 0 }}</div>
              <div class="stat-label">Tổng chunks</div>
            </div>
            <div class="stat-card">
              <div class="stat-number">{{ Object.keys(stats.categories || {}).length }}</div>
              <div class="stat-label">Danh mục</div>
            </div>
          </div>

          <!-- Documents List -->
          <div class="documents-list">
            <div class="list-header">
              <h3>Danh sách tài liệu</h3>
              <div class="search-box">
                <input
                  v-model="searchQuery"
                  placeholder="Tìm kiếm tài liệu..."
                  @input="searchDocuments"
                />
              </div>
            </div>

            <div v-if="loading" class="loading">
              <span class="spinner"></span>
              Đang tải...
            </div>

            <div v-else-if="documents.length === 0" class="empty-state">
              <el-icon size="48"><Document /></el-icon>
              <h3>Chưa có tài liệu nào</h3>
              <p>Hãy tải lên tài liệu đầu tiên để bắt đầu sử dụng hệ thống RAG</p>
            </div>

            <div v-else class="documents-grid">
              <div
                v-for="doc in documents"
                :key="doc.id"
                class="document-card"
              >
                <div class="document-header">
                  <el-icon><Document /></el-icon>
                  <div class="document-title">{{ doc.filename }}</div>
                </div>
                
                <div class="document-info">
                  <div class="info-item">
                    <span class="label">Danh mục:</span>
                    <span class="value">{{ doc.category || 'Chung' }}</span>
                  </div>
                  <div class="info-item">
                    <span class="label">Chunks:</span>
                    <span class="value">{{ doc.chunk_count }}</span>
                  </div>
                  <div class="info-item">
                    <span class="label">Ngày tải:</span>
                    <span class="value">{{ formatDate(doc.uploaded_at) }}</span>
                  </div>
                </div>

                <div class="document-actions">
                  <button 
                    v-if="canDelete(doc)" 
                    @click="deleteDocument(doc.id)" 
                    class="btn-delete"
                  >
                    <el-icon><Delete /></el-icon>
                    Xóa
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Upload Dialog -->
    <el-dialog
      v-model="showUploadDialog"
      title="Tải lên tài liệu"
      width="500px"
    >
      <div class="upload-form">
        <div class="form-group">
          <label>Chọn file (.txt)</label>
          <input
            ref="fileInput"
            type="file"
            accept=".txt"
            @change="handleFileSelect"
            class="file-input"
          />
        </div>

        <div class="form-group">
          <label>Danh mục</label>
          <select v-model="uploadForm.category" class="form-input">
            <option value="admission">Tuyển sinh</option>
            <option value="academic">Học vụ</option>
            <option value="general">Thông tin chung</option>
          </select>
        </div>

        <div v-if="selectedFile" class="file-info">
          <p><strong>File đã chọn:</strong> {{ selectedFile.name }}</p>
          <p><strong>Kích thước:</strong> {{ formatFileSize(selectedFile.size) }}</p>
        </div>
      </div>

      <template #footer>
        <div class="dialog-footer">
          <button @click="showUploadDialog = false" class="btn-cancel">
            Hủy
          </button>
          <button 
            @click="uploadDocument" 
            :disabled="!selectedFile || uploading"
            class="btn-upload-confirm"
          >
            <span v-if="uploading" class="spinner"></span>
            {{ uploading ? 'Đang tải lên...' : 'Tải lên' }}
          </button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import { ref, reactive, onMounted, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '@/services/api'

export default {
  name: 'Documents',
  setup() {
    const router = useRouter()
    const authStore = useAuthStore()
    
    const documents = ref([])
    const stats = ref({})
    const loading = ref(false)
    const uploading = ref(false)
    const showUploadDialog = ref(false)
    const searchQuery = ref('')
    const selectedFile = ref(null)
    const fileInput = ref(null)
    
    const uploadForm = reactive({
      category: ''
    })
    
    // Check permissions
    const canUpload = computed(() => {
      return authStore.user && ['teacher', 'admin'].includes(authStore.user.role)
    })
    
    const canDelete = (doc) => {
      if (!authStore.user) return false
      return authStore.user.role === 'admin' || doc.uploaded_by === authStore.user.id
    }
    
    const userRole = computed(() => {
      return authStore.user?.role || null
    })
    
    const isUserLoaded = computed(() => {
      return authStore.user !== null && authStore.user !== undefined
    })
    
    const fetchDocuments = async () => {
      loading.value = true
      try {
        const response = await api.get('/documents/list')
        documents.value = response.data
      } catch (error) {
        ElMessage.error('Không thể tải danh sách tài liệu')
      } finally {
        loading.value = false
      }
    }
    
    const fetchStats = async () => {
      try {
        const response = await api.get('/documents/stats')
        stats.value = response.data
      } catch (error) {
        console.error('Không thể tải thống kê')
      }
    }
    
    const handleFileSelect = (event) => {
      const file = event.target.files[0]
      if (file && file.type === 'text/plain') {
        selectedFile.value = file
      } else {
        ElMessage.error('Vui lòng chọn file .txt')
        event.target.value = ''
      }
    }
    
    const uploadDocument = async () => {
      if (!selectedFile.value) return
      
      uploading.value = true
      const formData = new FormData()
      formData.append('file', selectedFile.value)
      if (uploadForm.category) {
        formData.append('category', uploadForm.category)
      }
      
      try {
        await api.post('/documents/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        
        ElMessage.success('Tải lên tài liệu thành công!')
        showUploadDialog.value = false
        selectedFile.value = null
        uploadForm.category = ''
        if (fileInput.value) {
          fileInput.value.value = ''
        }
        
        await fetchDocuments()
        await fetchStats()
      } catch (error) {
        ElMessage.error('Tải lên tài liệu thất bại')
      } finally {
        uploading.value = false
      }
    }
    
    const deleteDocument = async (docId) => {
      try {
        await ElMessageBox.confirm(
          'Bạn có chắc chắn muốn xóa tài liệu này?',
          'Xác nhận xóa',
          {
            confirmButtonText: 'Xóa',
            cancelButtonText: 'Hủy',
            type: 'warning'
          }
        )
        
        await api.delete(`/documents/${docId}`)
        ElMessage.success('Xóa tài liệu thành công!')
        await fetchDocuments()
        await fetchStats()
      } catch (error) {
        if (error !== 'cancel') {
          ElMessage.error('Xóa tài liệu thất bại')
        }
      }
    }
    
    const searchDocuments = async () => {
      if (!searchQuery.value.trim()) {
        await fetchDocuments()
        return
      }
      
      try {
        const response = await api.post('/documents/search', {
          query: searchQuery.value,
          limit: 50
        })
        // Note: This would need to be implemented in the backend
        // For now, we'll just filter the existing documents
        documents.value = documents.value.filter(doc => 
          doc.filename.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
          (doc.category && doc.category.toLowerCase().includes(searchQuery.value.toLowerCase()))
        )
      } catch (error) {
        ElMessage.error('Tìm kiếm thất bại')
      }
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
    
    const formatFileSize = (bytes) => {
      if (bytes === 0) return '0 Bytes'
      const k = 1024
      const sizes = ['Bytes', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    }
    
    const logout = () => {
      authStore.logout()
      router.push('/login')
    }
    
    onMounted(async () => {
      // Đảm bảo user được load từ localStorage
      if (authStore.token && !authStore.user) {
        await authStore.initializeAuth()
      }
      
      if (isUserLoaded.value) {
        fetchDocuments()
        fetchStats()
      }
    })
    
    // Watch for user loading
    watch(isUserLoaded, (newValue) => {
      if (newValue) {
        fetchDocuments()
        fetchStats()
      }
    })
    
    return {
      documents,
      stats,
      loading,
      uploading,
      showUploadDialog,
      searchQuery,
      selectedFile,
      fileInput,
      uploadForm,
      fetchDocuments,
      handleFileSelect,
      uploadDocument,
      deleteDocument,
      searchDocuments,
      formatDate,
      formatFileSize,
      logout,
      canUpload,
      canDelete,
      userRole,
      isUserLoaded
    }
  }
}
</script>

<style scoped>
.documents-layout {
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
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

.documents-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
}

.documents-header h2 {
  color: #2c3e50;
  font-size: 28px;
  font-weight: 600;
}

.btn-upload {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.3s ease;
}

.btn-upload:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.upload-notice {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 8px;
  color: #856404;
  font-size: 14px;
}

.upload-notice .el-icon {
  color: #f39c12;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 32px;
}

.stat-card {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  text-align: center;
}

.stat-number {
  font-size: 32px;
  font-weight: 700;
  color: #667eea;
  margin-bottom: 8px;
}

.stat-label {
  color: #6c757d;
  font-weight: 500;
}

.documents-list {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.list-header h3 {
  color: #2c3e50;
  font-size: 20px;
  font-weight: 600;
}

.search-box input {
  padding: 12px 16px;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  font-size: 16px;
  width: 300px;
  transition: border-color 0.3s ease;
}

.search-box input:focus {
  outline: none;
  border-color: #667eea;
}

.loading {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
  color: #6c757d;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #6c757d;
}

.empty-state h3 {
  margin: 16px 0 8px 0;
  color: #2c3e50;
}

.documents-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
}

.document-card {
  border: 1px solid #e1e5e9;
  border-radius: 12px;
  padding: 20px;
  transition: all 0.3s ease;
}

.document-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.document-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.document-title {
  font-weight: 600;
  color: #2c3e50;
  font-size: 16px;
}

.document-info {
  margin-bottom: 16px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
}

.info-item .label {
  color: #6c757d;
}

.info-item .value {
  font-weight: 500;
  color: #2c3e50;
}

.document-actions {
  display: flex;
  justify-content: flex-end;
}

.btn-delete {
  background: #e74c3c;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: background-color 0.3s ease;
}

.btn-delete:hover {
  background: #c0392b;
}

.upload-form {
  padding: 20px 0;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: #2c3e50;
}

.file-input {
  width: 100%;
  padding: 12px;
  border: 2px dashed #e1e5e9;
  border-radius: 8px;
  background: #f8f9fa;
  cursor: pointer;
}

.form-input {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  font-size: 16px;
  transition: border-color 0.3s ease;
}

.form-input:focus {
  outline: none;
  border-color: #667eea;
}

.file-info {
  background: #f8f9fa;
  padding: 16px;
  border-radius: 8px;
  margin-top: 16px;
}

.file-info p {
  margin: 4px 0;
  font-size: 14px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.btn-cancel {
  background: #6c757d;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: background-color 0.3s ease;
}

.btn-cancel:hover {
  background: #5a6268;
}

.btn-upload-confirm {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.3s ease;
}

.btn-upload-confirm:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.btn-upload-confirm:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid #f3f3f3;
  border-top: 2px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style> 