import { validateSessionInfo } from '../components/chat/ChatContainer';

declare global {
  function describe(name: string, fn: () => void): void;
  function test(name: string, fn: () => void): void;
  namespace jest {
    interface Matchers<R> {
      toBeDefined(): R;
      toBe(expected: any): R;
      toThrow(expected?: string | RegExp): R;
      toEqual(expected: any): R;
    }
  }
  function expect<T>(actual: T): jest.Matchers<T>;
}

describe('Frontend Session Validation', () => {
  test('should validate complete session data', () => {
    const validSession = {
      session_id: 'test-session-123',
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T01:00:00Z',
      message_count: 5,
      user_id: 'user-456',
      title: 'Test Session',
      messages: [
        {
          id: 'msg-1',
          role: 'user',
          content: 'Hello',
          timestamp: '2024-01-01T00:00:00Z'
        }
      ],
      is_active: true
    };

    const result = validateSessionInfo(validSession);
    expect(result).toBeDefined();
    expect(result.session_id).toBe('test-session-123');
    expect(result.message_count).toBe(5);
    expect(result.is_active).toBe(true);
  });

  test('should handle minimal session data', () => {
    const minimalSession = {
      session_id: 'minimal-test'
    };

    const result = validateSessionInfo(minimalSession);
    expect(result).toBeDefined();
    expect(result.session_id).toBe('minimal-test');
    expect(result.message_count).toBe(0);
    expect(result.is_active).toBe(true);
  });

  test('should throw error for invalid session data', () => {
    const invalidSessions = [
      null,
      undefined,
      {},
      { invalid: 'data' },
      { session_id: null },
      { session_id: 123 }
    ];

    invalidSessions.forEach(session => {
      expect(() => validateSessionInfo(session)).toThrow();
    });
  });

  test('should handle array fields correctly', () => {
    const sessionWithMessages = {
      session_id: 'test-with-messages',
      messages: [
        { id: '1', role: 'user', content: 'Test', timestamp: '2024-01-01T00:00:00Z' },
        { id: '2', role: 'assistant', content: 'Response', timestamp: '2024-01-01T00:01:00Z' }
      ]
    };

    const result = validateSessionInfo(sessionWithMessages);
    expect(result.messages).toBeDefined();
    expect(result.messages?.length).toBe(2);
  });

  test('should set default values for optional fields', () => {
    const basicSession = {
      session_id: 'basic-test'
    };

    const result = validateSessionInfo(basicSession);
    expect(result.message_count).toBe(0);
    expect(result.is_active).toBe(true);
    expect(result.messages).toEqual([]);
  });
});
