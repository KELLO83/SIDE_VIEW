import os
import logging
import subprocess
from typing import Optional

class MPSManager:
    """NVIDIA Multi-Process Service (MPS) 관리 클래스"""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.is_enabled = False
        self.logger = logging.getLogger(__name__)
        
    def enable(self) -> bool:
        try:
            password = "rr"
            cmd = f'echo "{password}" | sudo -S nvidia-smi -i {self.gpu_id} -c 3'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Exclusive 모드 활성화 실패: {result.stderr}")
                
            cmd = f'echo "{password}" | sudo -S nvidia-cuda-mps-control -d'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)            
            self.logger.info("MPS 활성화 성공")
            
            return True
        except Exception as e:
            self.logger.error(f"MPS 활성화 실패: {str(e)}")
            return False
        
    def disable(self) -> bool:
        try:
            if not self.is_enabled:
                return True
                
            os.system('echo quit | nvidia-cuda-mps-control')

            os.system(f'nvidia-smi -i {self.gpu_id} -c 0')
            
            self.is_enabled = False
            return True
            
        except Exception as e:
            self.logger.error(f"MPS 비활성화 실패: {str(e)}")
            return False
    
    def check_status(self) -> bool:
        """MPS 서버 상태 확인"""
        try:
            result = subprocess.run(
                'echo "query" | nvidia-cuda-mps-control',
                shell=True,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        
        except Exception as e:
            self.logger.error(f"MPS 상태 확인 실패: {str(e)}")
            return False
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.disable()