import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Optional
import numpy as np

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

@dataclass
class BCIConfig:
    """Configuration for BrainFlow BCI parameters"""
    board_id: int = BoardIds.SYNTHETIC_BOARD  # Replace with your board ID
    serial_port: str = ""  # COM port for your device
    ip_address: str = ""   # IP address (if needed)
    command_threshold: float = 0.5
    window_size: int = 256  # Data points to analyze at once
    sampling_rate: int = 250  # Hz, adjust based on your device

@dataclass
class RobotConfig:
    """Configuration for Go2 robot parameters"""
    connection_method: WebRTCConnectionMethod = WebRTCConnectionMethod.LocalSTA
    ip: str = "192.168.8.181"
    move_speed: float = 0.5
    move_duration: float = 2.0

class BrainFlowRobotController:
    def __init__(self, bci_config: BCIConfig, robot_config: RobotConfig):
        self.bci_config = bci_config
        self.robot_config = robot_config
        self.board = None
        self.robot_conn: Optional[Go2WebRTCConnection] = None
        self.running = False
        self.concentration_model = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Enable BrainFlow logger
        BoardShim.enable_dev_board_logger()

    async def setup_bci(self):
        """Initialize BCI connection and processing"""
        try:
            # Setup BrainFlow parameters
            params = BrainFlowInputParams()
            params.serial_port = self.bci_config.serial_port
            params.ip_address = self.bci_config.ip_address
            
            # Initialize board
            self.board = BoardShim(self.bci_config.board_id, params)
            self.board.prepare_session()
            self.board.start_stream()
            
            # Initialize concentration detection model
            self.concentration_model = MLModel(BrainFlowMetrics.CONCENTRATION.value)
            self.concentration_model.prepare()
            
            self.logger.info("BCI setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"BCI setup failed: {str(e)}")
            raise

    async def setup_robot(self):
        """Initialize robot connection"""
        try:
            self.robot_conn = Go2WebRTCConnection(
                self.robot_config.connection_method,
                ip=self.robot_config.ip
            )
            await self.robot_conn.connect()
            await self._set_motion_mode("normal")
            self.logger.info("Robot connection established")
            
        except Exception as e:
            self.logger.error(f"Robot setup failed: {str(e)}")
            raise

    async def _set_motion_mode(self, mode: str):
        """Set robot motion mode"""
        await self.robot_conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {
                "api_id": 1002,
                "parameter": {"name": mode}
            }
        )
        await asyncio.sleep(2)

    async def move_forward(self):
        """Execute forward movement"""
        try:
            await self.robot_conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {
                        "x": self.robot_config.move_speed,
                        "y": 0,
                        "z": 0
                    }
                }
            )
            self.logger.info("Moving forward")
            await asyncio.sleep(self.robot_config.move_duration)
            
            # Stop movement
            await self.robot_conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {"x": 0, "y": 0, "z": 0}
                }
            )
            
        except Exception as e:
            self.logger.error(f"Movement execution failed: {str(e)}")

    def process_brain_data(self, data):
        """Process brain data and detect concentration levels"""
        try:
            # Get the EEG channels for your specific board
            eeg_channels = BoardShim.get_eeg_channels(self.bci_config.board_id)
            
            # Extract EEG data
            eeg_data = data[eeg_channels, :]
            
            # Apply bandpass filter (typical for concentration: 8-30 Hz)
            for channel in eeg_data:
                DataFilter.perform_bandpass(
                    channel,
                    self.bci_config.sampling_rate,
                    8.0,  # Start frequency
                    30.0, # Stop frequency
                    3,    # Order
                    brainflow.FilterTypes.BUTTERWORTH.value,
                    0
                )
            
            # Get concentration score
            bands = DataFilter.get_avg_band_powers(eeg_data, self.bci_config.sampling_rate, True)
            feature_vector = np.array(bands[0])
            concentration_score = self.concentration_model.predict(feature_vector)
            
            return concentration_score
            
        except Exception as e:
            self.logger.error(f"Error processing brain data: {str(e)}")
            return 0.0

    async def run(self):
        """Main control loop"""
        try:
            await self.setup_bci()
            await self.setup_robot()
            
            self.running = True
            self.logger.info("Starting brain activity monitoring...")
            
            while self.running:
                # Get data with window size
                data = self.board.get_current_board_data(self.bci_config.window_size)
                
                if data.size > 0:
                    concentration = self.process_brain_data(data)
                    
                    if concentration >= self.bci_config.command_threshold:
                        self.logger.info(f"High concentration detected: {concentration}")
                        await self.move_forward()
                
                await asyncio.sleep(0.1)  # Prevent CPU overload
                
        except Exception as e:
            self.logger.error(f"Runtime error: {str(e)}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        if self.board:
            self.board.stop_stream()
            self.board.release_session()
            
        if self.concentration_model:
            self.concentration_model.release()
            
        if self.robot_conn:
            await self.robot_conn.disconnect()
            
        self.logger.info("Cleanup completed")

async def main():
    # Configure your parameters
    bci_config = BCIConfig(
        board_id=BoardIds.SYNTHETIC_BOARD,  # Replace with your board
        serial_port="COM3",  # Replace with your port
        command_threshold=0.6
    )
    
    robot_config = RobotConfig(
        connection_method=WebRTCConnectionMethod.LocalSTA,
        ip="192.168.8.181",
        move_speed=0.5,
        move_duration=2.0
    )
    
    controller = BrainFlowRobotController(bci_config, robot_config)
    
    try:
        await controller.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        await controller.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 