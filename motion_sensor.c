#include <gpiod.h>
#include <stdio.h>
#include <unistd.h>

// Jetson Orin Nano — header pin 3 = I2C1_SDA (SoC PCC.01) = gpiochip0 line 13
// Bypasses the TXB0108 level shifter (I2C pins go direct to SoC).
// Requires unbinding the I2C driver first:
//   echo c250000.i2c | sudo tee /sys/bus/platform/drivers/tegra-i2c/unbind
#define GPIO_CHIP "gpiochip0"
#define GPIO_LINE 105

int main(void)
{
    struct gpiod_chip *chip = gpiod_chip_open_by_name(GPIO_CHIP);
    if (!chip) {
        perror("gpiod_chip_open_by_name");
        return 1;
    }

    struct gpiod_line *line = gpiod_chip_get_line(chip, GPIO_LINE);
    if (!line) {
        perror("gpiod_chip_get_line");
        gpiod_chip_close(chip);
        return 1;
    }

    if (gpiod_line_request_input(line, "motion_sensor") < 0) {
        perror("gpiod_line_request_input");
        gpiod_chip_close(chip);
        return 1;
    }

    while (1) {
        int val = gpiod_line_get_value(line);
        if (val < 0) {
            perror("gpiod_line_get_value");
            break;
        }
        if (val == 0) {
            printf("Somebody is in this area!\n");
        } else {
            printf("No one!\n");
        }
        usleep(10000); // 10 ms
    }

    gpiod_line_release(line);
    gpiod_chip_close(chip);
    return 0;
}
