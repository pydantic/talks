import logfire

APP_NAME = "example"

logfire.configure(
    service_name='cli',
    service_version='0.1.0',
)

def main():
    logfire.info(f"Launching app: {APP_NAME}")


if __name__ == '__main__':
    main()
