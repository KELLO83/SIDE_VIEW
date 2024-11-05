import cv2
class SitRecognition():
    def __init__(self):
        # 좌석 좌표 초기화
        self._init_seat_coordinates()
        
        # 이미지 로드 및 전처리
        self._init_images()
        
        # 시뮬레이션 좌석 좌표 설정  
        self._init_simulation_coordinates()
        
        # 전체 좌석 수 계산
        self.all_sits = (len(self.sits_simul_door1) + len(self.sits_simul_door2) + 
                        len(self.sits_simul_door3) + len(self.sits_simul_bus1_under) + 
                        len(self.sits_simul_bus2_in) + len(self.sits_bus2_out))

    def _init_seat_coordinates(self):
        """실제 좌석 좌표 초기화"""
        self.sits_door1 = [[(409, 247), (501, 338)], [(376, 141), (471, 227)], 
                          [(245, 249), (340, 337)], [(239, 173), (343, 245)], 
                          [(94, 248), (187, 359)], [(101, 157), (199, 243)]]
        
        self.sits_bus1_under = [[(421, 180), (586, 336)], [(170, 375), (264, 557)],
                               [(14, 382), (156, 484)], [(453, 412), (597, 598)]]
        
        self.sits_bus1_out = [[(303, 309), (411, 489)], [(419, 298), (508, 490)],
                             [(2, 263), (166, 430)]]
        
        self.sits_door2 = [[(376, 170), (560, 338)], [(159, 136), (357, 327)]]
        
        self.sits_door3 = [[(130, 328), (223, 492)], [(15, 316), (104, 485)],
                          [(250, 182), (357, 270)], [(244, 97), (366, 179)],
                          [(576, 341), (595, 386)], [(488, 334), (593, 478)],
                          [(357, 342), (482, 525)], [(84, 186), (213, 279)],
                          [(89, 96), (210, 182)]]
        
        self.sits_bus2_in = [[(131, 392), (210, 572)], [(2, 389), (105, 552)],
                            [(459, 399), (586, 534)], [(348, 403), (449, 559)]]
        
        self.sits_bus2_out = [[(389, 217), (453, 382)], [(485, 217), (560, 393)],
                             [(61, 221), (153, 387)], [(157, 216), (250, 387)],
                             [(396, 117), (493, 169)], [(383, 65), (503, 101)],
                             [(376, 21), (494, 60)], [(132, 108), (225, 164)],
                             [(154, 66), (231, 104)], [(169, 26), (244, 64)]]
        
        self.sits_bus1_under_cam1 = [[(384, 170), (495, 251)], [(407, 262), (529, 340)],
                                    [(254, 188), (368, 265)], [(232, 272), (374, 343)]]

    def _init_images(self):
        """이미지 로드 및 전처리"""
        self.full_sit = cv2.imread("./simulation_img/full_sit.png")
        self.empty_sit = cv2.imread("./simulation_img/empty_sit.png") 
        self.brt_simul = cv2.imread("./simulation_img/brt_simulation.png")
        
        self.sit_h, self.sit_w = self.full_sit.shape[:2]
        self.full_sit_simul = cv2.resize(self.full_sit, (int(self.sit_w*0.2), int(self.sit_h*0.2)))
        self.new_sit_h, self.new_sit_w = self.full_sit_simul.shape[:2]
        self.empty_sit_simul = cv2.cvtColor(self.full_sit_simul, cv2.COLOR_BGR2GRAY)
        self.empty_sit_simul = cv2.cvtColor(self.empty_sit_simul, cv2.COLOR_GRAY2BGR)

    def _init_simulation_coordinates(self):
        """시뮬레이션 좌석 좌표 설정"""
        base_x1, base_x2 = 115, 800
        w, h = self.new_sit_w, self.new_sit_h
        gap = 15
        
        self.sits_simul_door1 = [(base_x1, 140), (base_x1, 175),
                                (base_x1+w+gap, 140), (base_x1+w+gap, 175),
                                (base_x1+w*2+gap*2, 140), (base_x1+w*2+gap*2, 175)]
        
        self.sits_simul_bus1_under = [(base_x1+w*2+gap*2 , 35), (base_x1+w*3+gap*3 , 140),
                                     (base_x1+w*3+gap*3 , 175), (base_x1+w*3+gap*3 , 35)]
        
        self.sits_simul_bus1_out = [(base_x1+w*4+gap*4, 140), (base_x1+w*4+gap*4, 175),
                                   (base_x1+w*4+gap*4, 35)]
        
        self.sits_simul_door2 = [(base_x1+w*5+gap*5, 175), (base_x1+w*6+gap*6, 175)]
        
        self.sits_simul_door3 = [(base_x2, 140), (base_x2, 175),
                                (base_x2+w+gap, 140), (base_x2+w+gap, 175),
                                (775, 35), (775+h, 35), (775+h*2, 35),
                                (base_x2+w*2+gap*2, 140), (base_x2+w*2+gap*2, 175)]
        
        self.sits_simul_bus2_in = [(base_x2+w*3+gap*3, y) for y in [140, 175, 35, 70]]
        
        self.sits_simul_bus2_out = [(base_x2+w*4+gap*4, y) for y in [140, 175, 35, 70]] + \
                                  [(base_x2+w*5+gap*5+h*i, y) for i in range(3) for y in [160, 35]]

    def run(self):
        pass
    
    def visualize_seats(self, seat_status: dict[str, dict[str, bool]]) -> cv2.Mat:
        """좌석 상태 시각화
        args:
            seat_status: {
                1: {'LEFT': 0, 'RIGHT': 0},
                2: {'LEFT': 0, 'RIGHT': 0},
                3: {'LEFT': 0, 'RIGHT': 0},
                4: {'LEFT': 0, 'RIGHT': 0},
                'side_seat': {'seat9': 0, 'seat10': 0}
            }
            or 
            {
                'row1': {'LEFT': 0, 'RIGHT': 0},
                'row2': {'LEFT': 0, 'RIGHT': 0},
                'row3': {'LEFT': 0, 'RIGHT': 0},
                'row4': {'LEFT': 0, 'RIGHT': 0},
                'side_seat': {'seat9': 0, 'seat10': 0}
            }
        """
        
        new_seat_status = {}
        result_img = self.brt_simul.copy()
        for key , value in seat_status.items(): # int형일떄 조정작업
            if isinstance(key , int):
                new_key = f"row{key}"
                new_seat_status[new_key] = value
            else:
                new_seat_status[key] = value
        seat_status = new_seat_status.copy()
        
        print("DEBUG: ", seat_status)    
                
        # 모든 좌석 좌표 리스트
        all_seats = {
            'door1': self.sits_simul_door1,
            'bus1_under': self.sits_simul_bus1_under,
            'bus1_out': self.sits_simul_bus1_out,
            'door2': self.sits_simul_door2,
            'door3': self.sits_simul_door3,
            'bus2_in': self.sits_simul_bus2_in,
            'bus2_out': self.sits_simul_bus2_out
        }
        
        
        # 각 좌석 위치에 이미지 오버레이 
        for section_name, coordinates in all_seats.items():
            for coord in coordinates:
                x, y = coord
                # 이미지가 화면을 벗어나지 않도록 경계 확인
                if y + self.new_sit_h <= result_img.shape[0] and x + self.new_sit_w <= result_img.shape[1]:
                    # 기본적으로 빈 좌석으로 설정
                    overlay_img = self.empty_sit_simul
                    
                    # seat_status에서 해당 좌석이 차있는지 확인
                    for row, positions in seat_status.items():
                        seat_coords = self.get_seat_coordinates(row, positions)
                        if coord in seat_coords:
                            overlay_img = self.full_sit_simul
                            break
                            
                    # 이미지 오버레이
                    result_img[y:y+self.new_sit_h, x:x+self.new_sit_w] = overlay_img

        return result_img

    def get_seat_coordinates(self, row: str, positions: dict) -> list:
        """특정 row의 좌석 좌표 반환"""
        seat_mapping = {
            'row1': {'LEFT': self.sits_simul_door1[1], 'RIGHT': self.sits_simul_door1[0]},
            'row2': {'LEFT': self.sits_simul_door1[3], 'RIGHT': self.sits_simul_door1[2]}, 
            'row3': {'LEFT': self.sits_simul_door1[5], 'RIGHT': self.sits_simul_door1[4]},
            'row4': {'LEFT': self.sits_simul_bus1_under[2], 'RIGHT': self.sits_simul_bus1_under[1]}
        }

        coordinates = []
        if row == 'side_seat':  
            if positions.get('seat9'):
                coordinates.append(self.sits_simul_bus1_under[0])
            if positions.get('seat10'):
                coordinates.append(self.sits_simul_bus1_under[3])
        elif row in seat_mapping:  # 일반 좌석 처리
            for side in ['LEFT', 'RIGHT']:
                if positions.get(side) or positions.get(side.lower()):
                    coordinates.append(seat_mapping[row][side])
        
        return coordinates

if __name__ == "__main__":
    # SitRecognition 인스턴스 생성
    sit_recognition = SitRecognition()
    
    # 테스트용 좌석 상태 데이터
    test_seat_status = {
        'row1': {'left': True, 'right': False},
        'row2': {'left': True, 'right': False},
        'row3': {'left': True, 'right': False},
        'row4': {'left': True, 'right': False},
        'side_seat': {'seat9': True, 'seat10': True}
    }
    
    # 시각화 실행
    result = sit_recognition.visualize_seats(test_seat_status)
    
    # 좌표 표시
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 0, 255)
    thickness = 1
    
    # 각 섹션별 좌표 표시
    sections = {
        'D1': sit_recognition.sits_simul_door1,
        'B1U': sit_recognition.sits_simul_bus1_under,
        'B1O': sit_recognition.sits_simul_bus1_out,
        'D2': sit_recognition.sits_simul_door2,
        'D3': sit_recognition.sits_simul_door3,
        'B2I': sit_recognition.sits_simul_bus2_in,
        'B2O': sit_recognition.sits_simul_bus2_out
    }
    import random
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(sections))]
    for idx, (section_name, coordinates) in enumerate(sections.items()):
        if section_name == 'D1' or section_name == 'B1U':
            color = colors[idx]
            for i, (x, y) in enumerate(coordinates):
                text = f'{section_name}_{i}:({x},{y})'
                cv2.putText(result, f"{i}", (x, y-5), font, font_scale, color, thickness)
                # 좌표 위치에 점 표시
                cv2.circle(result, (x, y), 2, color, -1)
    
    # 결과 출력
    cv2.namedWindow('Seat Visualization with Coordinates', cv2.WINDOW_NORMAL)
    cv2.imshow('Seat Visualization with Coordinates', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()